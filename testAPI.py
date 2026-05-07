import requests

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
  "Authorization": "Bearer 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",
  "Content-Type": "application/json"
}

data = {
  "model": "openai/gpt-4o-mini",
  "messages": [
    {"role": "user", "content": "What is 2+2?"}
  ]
}

response = requests.post(url, headers=headers, json=data)

print(response.status_code)
response = response.json()
print(response["choices"][0]["message"]["content"])