import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import AIMessage, SystemMessage, HumanMessage
import logging

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
logging.info(
    f"Initialized OpenAI model: {model.model_name} with temperature: {model.temperature}"
)

class Product(BaseModel):
    Response: str=Field(description="The response from the agent, including the output and any intermediate steps.")
    Confidence: float=Field(description="The confidence level of the price lookup.")

logging.info("Defined Product model with fields: Name, Price, DiscountTier, DiscountedPrice, Confidence")


@tool()
def get_product_price(query: str) -> float:
    """
    Look up the current price of a product matching the user's description.

    Use this tool whenever the user asks about the price of a single product,
    even if the product description is generic or incomplete.

    Examples of valid inputs:
    - "laptop"
    - "headphones"
    - "webcam"
    - "keyboard"
    - "mouse"

    The query should be passed exactly as the user provided it.
    Do not ask the user for a more specific product unless this tool cannot
    determine a matching product.

    Returns:
        The current price of the best matching product.

    Always use this tool before answering any single-product price question.
    Never estimate or invent a price yourself.
    """

    logging.info(f"Looking up product: {query}")

    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
        "monitor": 249.99,
        "mouse": 39.99,
        "webcam": 79.95,
        "speaker": 119.90,
        "printer": 199.50,
        "tablet": 499.00,
    }

    return prices.get(query.lower(), 0.0)


@tool
def get_product_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a single product price.

    Use this function only after obtaining the product's price from
    `get_product_price`.

    If discounts need to be applied to multiple products, prefer using
    `get_multiple_product_discounts` instead of calling this function
    repeatedly.

    Input:
        price:
            The product price returned by `get_product_price`.

        discount_tier:
            The discount tier to apply.

            Example:
                "gold"
                "silver"
                "bronze"

    Output:
        The discounted price as a float.

        Example:
            899.10

    Notes:
        - Never calculate discounts yourself.
        - Always pass the exact price returned by `get_product_price`.
        - Do not modify or round the input price before calling this tool.
        - If the discount tier is unknown, ask the user instead of assuming one.
    """
    logging.info(
        f"Calculating discount for price: {price}, discount tier: {discount_tier}"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23, "platinum": 30}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


@tool
def get_multiple_product_prices(products: list[str]) -> dict[str, float]:
    """
    Returns the current prices of multiple products.

    Use this function whenever the user asks:
    - "How much do these products cost?"
    - "Compare the prices of..."
    - "What's the price of A, B, and C?"
    - Any question that requires the prices of multiple products.

    Input:
        products: List of product names.

    Output:
        Dictionary where:
        - key = product name
        - value = product price as a float

    Example:
        get_multiple_product_prices([
            "laptop",
            "headphones"
        ])

        returns
        {
            "laptop": 999.0,
            "headphones": 1199.0
        }

    Always include every product requested by the user in the `products` list.
    Prefer this function over repeated single-product price lookups.
    """
    logging.info(f"Fetching prices for products: {products}")
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
        "monitor": 249.99,
        "mouse": 39.99,
        "webcam": 79.95,
        "speaker": 119.90,
        "printer": 199.50,
        "tablet": 499.00,
    }
    return {product: prices.get(product, 0) for product in products}


@tool
def get_multiple_product_discounts(
    prices: dict[str, float], discount_tiers: dict[str, str]
) -> dict[str, float]:
    """
    Calculate the discounted price for multiple products based on their
    original prices and assigned discount tiers.

    Use this function whenever the user asks for:
    - Discounted prices for multiple products.
    - The final price after applying discounts.
    - Price comparisons after discounts.
    - The total cost of multiple discounted products.

    Input:
        prices:
            Dictionary mapping each product name to its original price.

            Example:
        {
            "laptop": 999.0,
            "headphones": 1199.0
        }

        discount_tiers:
            Dictionary mapping each product name to its discount tier.

            Example:
                {
                    "laptop": "gold",
                    "headphones": "silver"
                }

            Each product in `prices` should have a corresponding discount tier.

    Output:
        Dictionary mapping each product name to its final price after applying
        the corresponding discount.

        Example:
            {
                "laptop": 899.10,
                "headphones": 1139.05
            }

    Notes:
        - Pass all products in a single call instead of invoking the function
          multiple times.
        - Key and pair value in lower case.
        - Use the exact product names as the dictionary keys in both inputs.
        - The returned values are the final prices after discounts have been
          applied.
        - This function applies the discount associated with each product's
          discount tier; callers do not need to calculate discounts manually.
    """
    logging.info(
        f"Calculating discounted prices for products: {list(prices.keys())} with discount tiers: {discount_tiers}"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23, "platinum": 30}
    final_prices = {}
    for product, price in prices.items():
        discount_tier = discount_tiers.get(product)
        if discount_tier:
            discount = discount_percentages.get(discount_tier, 0)
            final_prices[product] = round(price * (1 - discount / 100), 2)
        else:
            final_prices[product] = price
    return final_prices


def run_agent(query: str, system_prompt: str = None, tools: list = None) -> dict:
    """AI Agent that can call tools to answer questions.

    Args:
        query (str): The user query to be answered by the agent.
        system_prompt (str, optional): The system prompt for the agent. Defaults to None.
        tools (list, optional): The list of tools available to the agent. Defaults to None.
    returns:
        response (dict): The response from the agent, including the output and any intermediate steps.
    raises:
        Exception: If there is an error while running the agent.
    """
    try:
        logging.info(f"Running agent with query: {query}")
        system_message = SystemMessage(content=system_prompt)
        human_message = HumanMessage(content=query)
        agent = create_agent(model=model, tools=tools)
        result = agent.invoke({"messages": [system_message, human_message]})

        messages = result["messages"]
        logging.info(f"Agent returned {len(messages)} messages.")
        last_ai_message = [m for m in messages if isinstance(m, AIMessage) and m.content.strip() != ""]
        logging.info(f"After filtered {len(last_ai_message)} AI messages.")
        for msg in last_ai_message:
            logging.info(f"AI Message: {msg.content}")

        return last_ai_message[0].content
    except Exception as e:
        logging.error(f"Error running agent: {e}")
        raise e


if __name__ == "__main__":
    tools = [
        get_multiple_product_prices,
        get_multiple_product_discounts,
        get_product_price,
        get_product_discount,
    ]
    with open("system_prompt.txt", "r") as f:
        system_prompt = f.read()
    query = """Get the laptop price"""

    #  """list the following products prices laptop, headphones, webcam, keyboard, mouse 
    # and apply gold discount to each product and provide the final price after discount."""

    # "What is laptop price? after applied gold discount?"

    response = run_agent(query, system_prompt, tools)
    with open("output.txt", "a") as f:
        f.write(f"Query: {query}\n")
        f.write(f"Response: {response}\n")
