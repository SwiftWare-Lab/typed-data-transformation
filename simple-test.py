
import sys

def main():
    
    if len(sys.argv) < 4:
        print("Usage: python script.py <string1> <string2> <integer> [additional arguments]")
        sys.exit(1)

    string1 = sys.argv[1]
    string2 = sys.argv[2]
    
    try:
        integer = int(sys.argv[3])  
    except ValueError:
        print("Error: The third argument must be an integer.")
        sys.exit(1)

    
    additional_args = sys.argv[4:]  

    
    if additional_args:
        print("Additional arguments provided:")
        for arg in additional_args:
            print(arg)

    print(f"String 1: {string1}")
    print(f"String 2: {string2}")
    print(f"Integer: {integer}")

if __name__ == "__main__":
    main()





