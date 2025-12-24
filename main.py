from decouple import config


def main():
    print("Hello from langchain-hello-world!")
    api_key = config("OPENAI_API_KEY")
    print(api_key)


if __name__ == "__main__":
    main()
