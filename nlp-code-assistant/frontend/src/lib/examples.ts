export interface Example {
  name: string;
  code: string;
  original?: string;
}

export const EXAMPLES: Example[] = [
  {
    name: "Boundary check (with original)",
    original: "def is_adult(age):\n    return age >= 18\n",
    code: "def is_adult(age):\n    return age > 18\n",
  },
  {
    name: "Running total (with original)",
    original:
      "def total_price(items, tax_rate):\n    subtotal = 0\n    for item in items:\n        subtotal = subtotal + item.price * item.quantity\n    return subtotal * (1 + tax_rate)\n",
    code: "def total_price(items, tax_rate):\n    subtotal = 0\n    for item in items:\n        subtotal = subtotal - item.price * item.quantity\n    return subtotal * (1 + tax_rate)\n",
  },
  {
    name: "Clean helper",
    code: 'def chunk(seq, size):\n    """Split seq into lists of at most size items."""\n    if size <= 0:\n        raise ValueError("size must be positive")\n    return [seq[i:i + size] for i in range(0, len(seq), size)]\n',
  },
];
