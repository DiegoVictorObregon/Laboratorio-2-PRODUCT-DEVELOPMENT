from dotenv import load_dotenv
import os

load_dotenv()  # Asume que el archivo .env está en el mismo directorio que este script
test_var = os.getenv("TEST_VARIABLE")
print("Value of TEST_VARIABLE:", test_var)
