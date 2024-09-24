import requests
import argparse
import time
from getpass import getpass

config = {"apiKey": "AIzaSyDp0FpJO01nRWeW4ZWZHet8V_Gcvl0xLAc"} # Fine to be public.
create_user_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={config['apiKey']}"
login_user_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={config['apiKey']}"
verify_and_forgot_password_url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={config['apiKey']}"
delete_user_url = f"https://identitytoolkit.googleapis.com/v1/accounts:delete?key={config['apiKey']}"

def delete_user(token: str) -> dict:
    """Delete the user account."""
    data = { "idToken": token }
    response = requests.post(delete_user_url, json=data)
    return response.json()
    
def create_user(email: str, password: str) -> dict:
    """Create a new user account."""
    data = { "email": email, "password": password, "returnSecureToken": True, }
    response = requests.post(create_user_url, json=data)
    return response.json()

def login(email: str, password: str) -> dict:
    """Login the user."""
    data = { "email": email, "password": password, "returnSecureToken": True}
    response = requests.post(login_user_url, json=data)
    if response.status_code != 200:
        raise ValueError("Login failed. Please check your password and make sure your account exists.")
    return response.json()

def forgot_password(email: str) -> dict:
    """Send a password reset email."""
    data = { "requestType": "PASSWORD_RESET", "email": email}
    response = requests.post(verify_and_forgot_password_url, json=data)
    return response.json()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submit homework to autograder')
    parser.add_argument("--create-account", type=bool, help="Create a new account.", action = argparse.BooleanOptionalAction)
    parser.add_argument("--forgot-password", type=bool, help="Send a password reset email.", action = argparse.BooleanOptionalAction)
    parser.add_argument("--delete-account", type=bool, help="Delete your account.", action = argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.create_account:
        print("-*-"*20)
        print("Please note: your email MUST be your Northwestern email.")
        print("-*-"*20)
        username = input("Enter your email address:")
        password = getpass("Enter your password: ")
        repeat_password = getpass("Repeat your password: ")

        if password != repeat_password:
            print("Passwords do not match.")
            exit()

        response = create_user(username, password)
        if "error" in response.keys():
            print("ERROR: ", response["error"]["message"])
            exit()
        print("Account created! You may now login.")
        exit()

    if args.forgot_password:
        username = input("Enter your email address: ")
        response = forgot_password(username)
        print(response)
        exit()
    
    if args.delete_account:
        username = input("Enter your email address:")
        password = getpass("Enter your password: ")
        login_info = login(username, password)
        response = delete_user(login_info["idToken"])
        exit()
