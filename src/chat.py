from search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta (digite 'sair' para encerrar):\n")

    while True:
        try:
            pergunta = input("PERGUNTA: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o chat.")
            break

        if not pergunta:
            continue

        if pergunta.lower() in ("sair", "exit", "quit"):
            print("Encerrando o chat.")
            break

        try:
            resposta = chain(pergunta)
            print(f"RESPOSTA: {resposta}\n")
        except Exception as e:
            print(f"Erro ao processar a pergunta: {e}\n")


if __name__ == "__main__":
    main()
