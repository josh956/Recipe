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
    Inject custom CSS for metric styling and fixed bottom bar.
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
        </style>
        """,
        unsafe_allow_html=True
    )

def normalize_ingredient_name(name: str) -> str:
    """
    Normalize an ingredient name for matching.
    """
    name = name.replace("-", " ")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
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
    Return only the numeric part (plus 'g' if present) from a nutrition string.
    """
    match = re.search(r"(\d+(\.\d+)?)(g)?", nutrition_text)
    if match:
        number = match.group(1)
        is_g = match.group(3)
        return number + "g" if is_g else number
    else:
        return nutrition_text

def extract_nutrition_facts(summary):
    """
    Extract and parse bold nutrition info from the summary.
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
    for instr in recipe.get("analyzedInstructions", []):
        for step in instr.get("steps", []):
            for equip in step.get("equipment", []):
                equipment_set.add(equip["name"])
    return sorted(equipment_set)

def build_ingredient_original_map(extended_ingredients):
    """
    Map normalized ingredient names to their original text.
    """
    original_map = {}
    for ing in extended_ingredients:
        raw_name = ing.get("name", "")
        original_str = ing.get("original", "")
        norm = normalize_ingredient_name(raw_name)
        original_map[norm] = original_str
    return original_map

def calculate_nutrition_and_cost(ingredients_text, servings):
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
    instructions = recipe.get("analyzedInstructions", [])
    steps_text = []
    for instr in instructions:
        for step in instr.get("steps", []):
            step_number = step.get("number", 0)
            step_content = step.get("step", "")
            steps_text.append(f"Step {step_number}: {step_content}")
    full_instructions = "\n".join(steps_text)
    extended_ingredients = recipe.get("extendedIngredients", [])
    full_ingredients_text = "\n".join([ing["original"] for ing in extended_ingredients])
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
        raw_result = re.sub(r"^```[a-zA-Z]*\n?", "", raw_result)
        raw_result = re.sub(r"```$", "", raw_result).strip()
        data = json.loads(raw_result)
        return data
    except Exception as e:
        st.error(f"Error inferring step ingredient amounts: {str(e)}")
        return None

def analyze_health_of_meal(recipe):
    instructions = recipe.get("analyzedInstructions", [])
    steps_text = []
    for instr in instructions:
        for step in instr.get("steps", []):
            step_number = step.get("number", 0)
            step_content = step.get("step", "")
            steps_text.append(f"Step {step_number}: {step_content}")
    full_instructions = "\n".join(steps_text)
    extended_ingredients = recipe.get("extendedIngredients", [])
    full_ingredients_text = "\n".join([ing["original"] for ing in extended_ingredients])
    prompt = (
        "You are a health and nutrition expert. I have a recipe with the following ingredients:\n"
        f"{full_ingredients_text}\n\n"
        "And the following instructions:\n"
        f"{full_instructions}\n\n"
        "Please provide an in-depth analysis of the health aspects of this meal. "
        "Discuss which ingredients are particularly beneficial and why, and which might be less healthy "
        "or need moderation. Include potential impacts on health, vitamins, minerals, and overall nutritional balance. "
        "Return your analysis in plain text."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a health and nutrition expert."},
                {"role": "user", "content": prompt}
            ]
        )
        raw_result = response.choices[0].message.content.strip()
        raw_result = re.sub(r"^```[a-zA-Z]*\n?", "", raw_result)
        raw_result = re.sub(r"```$", "", raw_result).strip()
        return raw_result
    except Exception as e:
        st.error(f"Error analyzing health of meal: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Recipe Viewer", layout="wide")
    inject_custom_css()
    st.title("Recipe Viewer")
    
    recipe_url = st.text_input("Enter the Recipe URL:", "https://natashaskitchen.com/salmon-cakes-recipe/")
    if st.button("Fetch Recipe"):
        with st.spinner("Fetching recipe..."):
            recipe = fetch_recipe(recipe_url)
        if recipe:
            # Sidebar: Recipe Details & Nutrition Metrics
            st.sidebar.title("Recipe Details")
            st.sidebar.markdown(f"**Ready in:** {recipe.get('readyInMinutes', 'N/A')} minutes")
            st.sidebar.markdown(f"**Servings:** {recipe.get('servings', 'N/A')}")
            
            nutrition_facts = extract_nutrition_facts(recipe.get("summary", ""))
            if nutrition_facts:
                st.sidebar.markdown('<div class="my-metric-container">', unsafe_allow_html=True)
                st.sidebar.markdown('<div class="my-metric-title">Nutrition Facts (per serving)</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.sidebar.columns(3)
                col1.metric("Calories", nutrition_facts.get("Calories", "N/A"))
                col2.metric("Protein", nutrition_facts.get("Protein", "N/A"))
                col3.metric("Fat", nutrition_facts.get("Fat", "N/A"))
                st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
            if "extendedIngredients" in recipe:
                ingredients_list = [ing["original"] for ing in recipe["extendedIngredients"]]
                ingredients_text = "; ".join(ingredients_list)
                servings = recipe.get("servings", 1)
                original_map = build_ingredient_original_map(recipe["extendedIngredients"])
                with st.spinner("Calculating total nutrition and cost..."):
                    result = calculate_nutrition_and_cost(ingredients_text, servings)
                if result:
                    result = result.strip()
                    result = re.sub(r"^```[a-zA-Z]*\n?", "", result)
                    result = re.sub(r"```$", "", result).strip()
                    if result.startswith("{") and result.endswith("}"):
                        try:
                            nutrition_cost = json.loads(result)
                            if "total_cost" in nutrition_cost:
                                nutrition_cost["total_cost"] = f"${float(nutrition_cost['total_cost']):.2f}"
                            if "price_per_serving" in nutrition_cost:
                                nutrition_cost["price_per_serving"] = f"${float(nutrition_cost['price_per_serving']):.2f}"
                            st.sidebar.markdown('<div class="my-metric-container">', unsafe_allow_html=True)
                            st.sidebar.markdown('<div class="my-metric-title">Total Nutrition & Cost (Recipe)</div>', unsafe_allow_html=True)
                            row1_col1, row1_col2, row1_col3 = st.sidebar.columns(3)
                            row1_col1.metric("Calories", str(nutrition_cost.get("calories", "N/A")))
                            row1_col2.metric("Protein", str(nutrition_cost.get("protein", "N/A")) + "g")
                            row1_col3.metric("Fat", str(nutrition_cost.get("fat", "N/A")) + "g")
                            row2_col1, row2_col2 = st.sidebar.columns(2)
                            row2_col1.metric("Total Cost", nutrition_cost.get("total_cost", "N/A"))
                            row2_col2.metric("Price/Serving", nutrition_cost.get("price_per_serving", "N/A"))
                            st.sidebar.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.sidebar.error("Failed to parse nutrition and cost: " + str(e))
                    else:
                        st.sidebar.error("Unexpected GPT response: " + result)
            
            with st.spinner("Inferring step-level ingredient amounts..."):
                step_amounts_map = infer_step_ingredient_amounts(recipe)
            
            with st.spinner("Analyzing health & nutrition aspects..."):
                health_analysis = analyze_health_of_meal(recipe)
            
            # Main Content: Recipe Title, Image, Summary
            st.header(recipe.get("title", "Recipe Title"))
            st.image(recipe["image"], use_container_width=True)
            st.markdown("### Summary")
            st.markdown(recipe.get("summary", ""), unsafe_allow_html=True)
            st.markdown("---")
            
            # Tabs: Equipment, Ingredients, Recipe Steps, Health & Nutrition
            tab_equipment, tab_ingredients, tab_recipe, tab_health = st.tabs(
                ["Equipment", "Ingredients", "Recipe Steps", "Health & Nutrition"]
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
                    for ing in recipe["extendedIngredients"]:
                        st.markdown(f"• {ing['original']}", unsafe_allow_html=True)
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
                            if step.get("equipment"):
                                equip_names = ", ".join([equip["name"] for equip in step["equipment"]])
                                st.write("Equipment: " + equip_names)
                            if step_amounts_map and str(step_num) in step_amounts_map:
                                usage_dict = step_amounts_map[str(step_num)]
                                if usage_dict:
                                    usage_strings = [f"{ing}: {amt}" for ing, amt in usage_dict.items()]
                                    st.write("**Amounts:** " + ", ".join(usage_strings))
                            else:
                                if step.get("ingredients"):
                                    ing_names = ", ".join([ing["name"] for ing in step["ingredients"]])
                                    st.write("Ingredients for this step: " + ing_names)
                            st.markdown("---")
                else:
                    st.write("No instructions available.")
            
            with tab_health:
                st.header("Health & Nutrition Analysis")
                if health_analysis:
                    st.write(health_analysis)
                else:
                    st.write("No health analysis available.")

if __name__ == "__main__":
    main()

st.markdown(
    """
    <hr>
    <p style="text-align: center;">
    <b>Recipe Viewer</b> &copy; 2025<br>
    Developed by <a href="https://www.linkedin.com/in/josh-poresky956/" target="_blank">Josh Poresky</a><br><br>
    </p>
    """,
    unsafe_allow_html=True
)
