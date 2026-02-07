from src.ui.web_app import create_app


def main():
    app = create_app()
    # 开发默认端口 8000，可根据需要调整
    app.run(host="127.0.0.1", port=8000, debug=True)


if __name__ == "__main__":
    main()


