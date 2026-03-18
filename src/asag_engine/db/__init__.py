def init_db(auto_create: bool = False):
    if auto_create:
        print(
            "[db] AUTO_CREATE_TABLES is ignored. "
            "Shared lms.* and ai.* schemas are managed outside this service."
        )
