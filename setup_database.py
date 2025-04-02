from modules.database import Base, db_manager

def main():
    print("Creating database tables...")
    Base.metadata.create_all(db_manager.engine)
    print("Database tables created successfully!")

if __name__ == "__main__":
    main()