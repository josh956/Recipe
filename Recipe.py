import streamlit as st
import requests
import os
import pandas as pd
import json
import re
from openai import OpenAI

# Setup OpenAI API Key from environment or Streamlit secrets
OPENAI_API_KEY = os.getenv("General") if os.getenv("General") else st.secrets["General"]["key"]
RAPIDAPI_KEY = os.getenv("RapidAPI") if os.getenv("RapidAPI") else st.secrets["rapidapi"]["key"]
client = OpenAI(api_key=OPENAI_API_KEY)

def inject_custom_css():
    """
    Injects custom CSS to style the metric containers.
    Adjust background-color, border-radius, and other properties as desired.
    """
    st.markdown(
        """
        <style>
        /* Container for a group of metrics */
        .my-metric-container {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 10px 15px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        /* Title above the metrics */
        .my-metric-title {
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 10px;
        }
        /* Ensure spacing between columns in a row of metrics */
        div[data-testid="metric-container"] > div {
            margin-right: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def normalize_ingredient_name(name: str) -> str:
    """
    Convert 'all-purpose flour' -> 'all purpose flour', remove punctuation,
    lowercase, etc. to help match step ingredient names to the original map.
    """
    # Replace hyphens with spaces
    name = name.replace("-", " ")
    # Remove punctuation except for letters, numbers, underscores, spaces
    name = re.sub(r"[^\w\s]", "", name)
    # Convert multiple spaces to a single space
    name = re.sub(r"\s+", " ", name)
    # Strip leading/trailing spaces and lowercase
    return name.strip().lower()

def fetch_recipe(recipe_url):
    api_url = "https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/extract"
    querystring = {"url": recipe_url}
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
    }
    response = requests.get(api_url, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch recipe. Please check the URL and try again.")
        return None

def parse_nutrition_value(nutrition_text):
    """
    Given a string like "447 calories", "8g of protein", "38g of fat",
    return just the numeric part plus 'g' if present:
      - "447" for "447 calories"
      - "8g"  for "8g of protein"
      - "38g" for "38g of fat"
    """
    match = re.search(r"(\d+(\.\d+)?)(g)?", nutrition_text)
    if match:
        number = match.group(1)     # e.g. "447" or "8" or "38"
        is_g = match.group(3)      # 'g' or None
        if is_g:
            return number + "g"    # e.g. "8g"
        else:
            return number          # e.g. "447"
    else:
        return nutrition_text      # fallback if no match

def extract_nutrition_facts(summary):
    """
    Extract bold-text nutrition info from the summary (e.g. "447 calories", "8g of protein", "38g of fat")
    and parse them down to just numeric values or numeric + 'g'.
    """
    nutrition = {}
    bold_texts = re.findall(r"<b>(.*?)</b>", summary)
    for text in bold_texts:
        lower_text = text.lower()
        if "calories" in lower_text:
            nutrition["Calories"] = parse_nutrition_value(text)
        elif "protein" in lower_text:
            nutrition["Protein"] = parse_nutrition_value(text)
        elif "fat" in lower_text:
            nutrition["Fat"] = parse_nutrition_value(text)
    return nutrition

def extract_unique_equipment(recipe):
    equipment_set = set()
    for instruction in recipe.get("analyzedInstructions", []):
        for step in instruction.get("steps", []):
            for equip in step.get("equipment", []):
                equipment_set.add(equip["name"])
    return sorted(equipment_set)

def build_ingredient_original_map(extended_ingredients):
    """
    Create a lookup dictionary mapping the ingredient's *normalized* name
    to the exact "original" text (the same text shown in the Ingredients list).
    """
    original_map = {}
    for ing in extended_ingredients:
        raw_name = ing.get("name", "")
        original_str = ing.get("original", "")
        norm = normalize_ingredient_name(raw_name)
        original_map[norm] = original_str
    return original_map

def calculate_nutrition_and_cost(ingredients_text, servings):
    """
    Uses the OpenAI ChatCompletion API to calculate total nutrition facts, total cost, 
    and price per serving based on the provided ingredients (with quantities).
    The model returns only numbers in a JSON format with keys:
    'calories', 'protein', 'fat', 'total_cost', and 'price_per_serving'.
    """
    prompt = (
        f"Given the following ingredients with their quantities: {ingredients_text}, "
        "calculate the total nutrition facts (calories, protein in grams, fat in grams), "
        "the total cost (in USD) for the entire recipe, and the price per serving (in USD) "
        "assuming the recipe yields the provided number of servings. "
        "Return ONLY a valid JSON object with exactly the keys 'calories', 'protein', 'fat', "
        "'total_cost', and 'price_per_serving' with no additional text or formatting."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a nutrition and cost calculator."},
                {"role": "user", "content": prompt}
            ]
        )
        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        st.error("Error calculating nutrition and cost: " + str(e))
        return None

def infer_step_ingredient_amounts(recipe):
    """
    Makes an additional GPT call to infer how much of each ingredient is used in each step.
    Returns a dict of the form:
    {
      "1": {
         "all-purpose flour": "1 cup",
         "salt": "1 tsp"
      },
      "2": { ... },
      ...
    }
    or None if parsing fails.
    """

    # Gather the entire instructions text in a user-friendly way
    instructions = recipe.get("analyzedInstructions", [])
    steps_text = []
    for instr in instructions:
        for step in instr.get("steps", []):
            step_number = step.get("number", 0)
            step_content = step.get("step", "")
            steps_text.append(f"Step {step_number}: {step_content}")
    full_instructions = "\n".join(steps_text)

    # Gather the full ingredient list (with total amounts)
    extended_ingredients = recipe.get("extendedIngredients", [])
    full_ingredients_text = "\n".join([ing["original"] for ing in extended_ingredients])

    # Construct the prompt for GPT
    # We ask GPT to read the instructions, read the ingredient list, and estimate usage per step.
    prompt = (
        "You are a cooking assistant. I have a recipe with instructions and a list of ingredients. "
        "The instructions do not specify exactly how much of each ingredient is used in each step. "
        "Using your best judgment, infer approximate step-by-step ingredient amounts. "
        "Only list ingredients that actually appear in that step. "
        "Return a JSON object mapping each step number (as a string) to an object with "
        "'ingredient name' : 'approximate amount' pairs. "
        "If an ingredient is not used in a step, do not list it for that step.\n\n"
        f"Instructions:\n{full_instructions}\n\n"
        f"Ingredients (with total amounts):\n{full_ingredients_text}\n\n"
        "Return ONLY valid JSON, no code fences, no extra explanation."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a cooking assistant that infers step-level amounts."},
                {"role": "user", "content": prompt}
            ]
        )
        raw_result = response.choices[0].message.content.strip()

        # Remove triple backticks if present
        raw_result = re.sub(r"^```[a-zA-Z]*\n?", "", raw_result)
        raw_result = re.sub(r"```$", "", raw_result).strip()

        # Attempt to parse as JSON
        data = json.loads(raw_result)
        return data

    except Exception as e:
        st.error(f"Error inferring step ingredient amounts: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Delicious Recipe Viewer", layout="wide")

    # Inject custom CSS for nicer metric styling
    inject_custom_css()

    st.title("Delicious Recipe Viewer")

    # User input for the recipe URL
    recipe_url = st.text_input(
        "Enter the Recipe URL:",
        "https://natashaskitchen.com/salmon-cakes-recipe/"
    )

    if st.button("Fetch Recipe"):
        with st.spinner("Fetching recipe..."):
            recipe = fetch_recipe(recipe_url)
        
        if recipe:
            # Sidebar: Display quick info
            st.sidebar.title("Recipe Details")
            st.sidebar.image(recipe["image"], use_container_width=True)
            st.sidebar.markdown(f"**Ready in:** {recipe.get('readyInMinutes', 'N/A')} minutes")
            st.sidebar.markdown(f"**Servings:** {recipe.get('servings', 'N/A')}")
            st.sidebar.markdown(f"**Price per Serving:** ${recipe.get('pricePerServing', 'N/A')}")

            # Per-Serving Nutrition Facts
            nutrition_facts = extract_nutrition_facts(recipe.get("summary", ""))
            if nutrition_facts:
                # Container for the per-serving metrics
                st.sidebar.markdown('<div class="my-metric-container">', unsafe_allow_html=True)
                st.sidebar.markdown('<div class="my-metric-title">Nutrition Facts (per serving)</div>', unsafe_allow_html=True)

                col1, col2, col3 = st.sidebar.columns(3)
                col1.metric("Calories", nutrition_facts.get("Calories", "N/A"))
                col2.metric("Protein", nutrition_facts.get("Protein", "N/A"))
                col3.metric("Fat", nutrition_facts.get("Fat", "N/A"))

                st.sidebar.markdown('</div>', unsafe_allow_html=True)

            # Calculate total nutrition and cost using the ingredients list via OpenAI API
            if "extendedIngredients" in recipe:
                ingredients_list = [ing["original"] for ing in recipe["extendedIngredients"]]
                ingredients_text = "; ".join(ingredients_list)
                servings = recipe.get("servings", 1)  # default to 1 if not provided

                # Build a map of normalized name -> exact original text
                original_map = build_ingredient_original_map(recipe["extendedIngredients"])
                
                with st.spinner("Calculating total nutrition and cost..."):
                    result = calculate_nutrition_and_cost(ingredients_text, servings)
                
                if result:
                    # Remove code fences if present
                    result = result.strip()
                    result = re.sub(r"^```[a-zA-Z]*\n?", "", result)
                    result = re.sub(r"```$", "", result)
                    result = result.strip()

                    if result.startswith("{") and result.endswith("}"):
                        try:
                            nutrition_cost = json.loads(result)
                            # Round the cost columns to 2 decimals, format with $xx.xx
                            if "total_cost" in nutrition_cost:
                                nutrition_cost["total_cost"] = f"${float(nutrition_cost['total_cost']):.2f}"
                            if "price_per_serving" in nutrition_cost:
                                nutrition_cost["price_per_serving"] = f"${float(nutrition_cost['price_per_serving']):.2f}"

                            # Container for the total recipe metrics
                            st.sidebar.markdown('<div class="my-metric-container">', unsafe_allow_html=True)
                            st.sidebar.markdown('<div class="my-metric-title">Total Nutrition & Cost (Recipe)</div>', unsafe_allow_html=True)

                            # First row: Calories, Protein, Fat
                            row1_col1, row1_col2, row1_col3 = st.sidebar.columns(3)
                            row1_col1.metric("Calories", str(nutrition_cost.get("calories", "N/A")))
                            row1_col2.metric("Protein", str(nutrition_cost.get("protein", "N/A")) + "g")
                            row1_col3.metric("Fat", str(nutrition_cost.get("fat", "N/A")) + "g")

                            # Second row: Total Cost, Price/Serving
                            row2_col1, row2_col2 = st.sidebar.columns(2)
                            row2_col1.metric("Total Cost", nutrition_cost.get("total_cost", "N/A"))
                            row2_col2.metric("Price/Serving", nutrition_cost.get("price_per_serving", "N/A"))

                            st.sidebar.markdown('</div>', unsafe_allow_html=True)

                        except Exception as e:
                            st.sidebar.error("Failed to parse nutrition and cost: " + str(e))
                    else:
                        st.sidebar.error("Unexpected GPT response: " + result)

            # ---- NEW GPT CALL FOR STEP-LEVEL AMOUNTS ----
            step_amounts_map = None
            with st.spinner("Inferring step-level ingredient amounts..."):
                step_amounts_map = infer_step_ingredient_amounts(recipe)
            
            # Main content: Title, Image, and Summary
            st.header(recipe.get("title", "Recipe Title"))
            st.image(recipe["image"], use_container_width=True)
            st.markdown("### Summary")
            st.markdown(recipe.get("summary", ""), unsafe_allow_html=True)
            st.markdown("---")

            # Tabs for organized content: Equipment, Ingredients, and Recipe Steps
            tab_equipment, tab_ingredients, tab_recipe = st.tabs(
                ["Equipment", "Ingredients", "Recipe Steps"]
            )

            with tab_equipment:
                st.header("Equipment Needed")
                equipment = extract_unique_equipment(recipe)
                if equipment:
                    for item in equipment:
                        st.write(f"• {item}")
                else:
                    st.write("No equipment information available.")

            with tab_ingredients:
                st.header("Ingredients")
                if "extendedIngredients" in recipe:
                    for ingredient in recipe["extendedIngredients"]:
                        st.write(f"• {ingredient['original']}")
                else:
                    st.write("No ingredients information available.")

            with tab_recipe:
                st.header("Step-by-Step Instructions")
                instructions = recipe.get("analyzedInstructions", [])
                if instructions:
                    for instr in instructions:
                        for step in instr.get("steps", []):
                            step_num = step.get("number", 0)
                            st.markdown(f"**Step {step_num}:** {step['step']}")

                            # Show equipment
                            if step.get("equipment"):
                                equip_names = ", ".join([equip["name"] for equip in step["equipment"]])
                                st.write("Equipment: " + equip_names)

                            # If we got a step-level ingredient usage map from GPT
                            # we attempt to display GPT's guessed amounts for this step.
                            if step_amounts_map and str(step_num) in step_amounts_map:
                                # It's a dictionary of { ingredientName: "approximate amount" }
                                usage_dict = step_amounts_map[str(step_num)]
                                if usage_dict:
                                    # Format them as bullet points or a single line
                                    # E.g. "Flour: 1 cup, Salt: 1 tsp"
                                    usage_strings = []
                                    for ing_name, amt in usage_dict.items():
                                        usage_strings.append(f"{ing_name}: {amt}")
                                    usage_line = ", ".join(usage_strings)
                                    st.write("**GPT-Inferred Ingredient Amounts:** " + usage_line)
                            else:
                                # fallback: just show the ingredient names from Spoonacular
                                if step.get("ingredients"):
                                    ing_names = ", ".join([ing["name"] for ing in step["ingredients"]])
                                    st.write("Ingredients for this step: " + ing_names)

                            st.markdown("---")
                else:
                    st.write("No instructions available.")

if __name__ == "__main__":
    main()

# --- Footer ---
st.markdown(
    """
    <hr>
    <p style="text-align: center;">
    <b>CO₂ Emissions Calculator</b> &copy; 2025<br>
    Developed by <a href="https://www.linkedin.com/in/josh-poresky956/" target="_blank">Josh Poresky</a><br><br>
    </p>
    """,
    unsafe_allow_html=True
)
