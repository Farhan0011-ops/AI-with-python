def tester(givenstring="Too short")
    if len(givenstring) < 10:
        print(" the string is too short")
    else:
        print(givenstring)
def main():
    while True:
        UserInput = input("Write something (quit ends): ")
        if UserInput == "quit":
            break
        tester(UserInput)
main()
