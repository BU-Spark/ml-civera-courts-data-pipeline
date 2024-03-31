from transformers import AutoTokenizer, AutoModel


def load():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()
    model = model.eval()
    return tokenizer, model


def chat(tokenizer, model):
    while True:
        # prompt = "Use one or two word to summarize this description: 'Notice of transfer to court'"
        print("User: ", end="")
        prompt = input()
        response, history = model.chat(tokenizer, prompt, history=[])
        print("GLM: ", end="")
        print(response)


def get_response(tokenizer, model, prompt, if_print=False):
    response, history = model.chat(tokenizer, prompt, history=[])
    if not if_print:
        return response
    print(f"\nUser: {prompt}\nGLM: {response}\n")
    return response


if __name__ == '__main__':

    chat(*load())

