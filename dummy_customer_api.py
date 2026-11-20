# file: dummy_customer_api.py
from flask import Flask, request, jsonify
import random

from sample_orders import ORDERS

app = Flask(__name__)

# simulate messy unstructured text responses
@app.route("/api/orders", methods=["GET"])
def get_orders():
    """
    Returns orders as messy text. In real life, customers
    would have unpredictable formatting. The AI must parse it.
    """
    limit = request.args.get("limit", default=len(ORDERS), type=int)
    sample = random.sample(ORDERS, min(limit, len(ORDERS)))

    return jsonify({
        "status": "ok",
        "raw_orders": sample
    })


@app.route("/api/order/<order_id>", methods=["GET"])
def get_order_by_id(order_id):
    """
    Fetch a single order by scanning the text.
    """
    for text in ORDERS:
        if order_id in text:
            return jsonify({
                "status": "ok",
                "raw_order": text
            })

    return jsonify({"status": "not_found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
