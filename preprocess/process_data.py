try:
    from .process_shapenet import configure_logging, main
except ImportError:
    from process_shapenet import configure_logging, main


if __name__ == "__main__":
    configure_logging()
    main()
