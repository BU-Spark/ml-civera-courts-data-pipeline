import google.generativeai as genai


with open('cfg/gemini_api_key.yml', 'r') as file:
    lines = file.readlines()
    API_KEY = lines[0].strip()
    print(API_KEY)


genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')


def get_response(prompt, if_print=False):
    response = model.generate_content(prompt)

    if if_print:
        print(response.text)

    return response.text


