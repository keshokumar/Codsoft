import random

Ai="O"
User="X"

def create_board():
    board = [["_" for _ in range(3)] for _ in range(3)]
    return board
def print_board(board):
    for row in board:
        print("|".join(row))
        print("_"*9)

def check_win(board,player):
    winning_com=[
        [(0, 0), (0, 1), (0, 2)],  
        [(1, 0), (1, 1), (1, 2)],  
        [(2, 0), (2, 1), (2, 2)], 
        [(0, 0), (1, 0), (2, 0)],  
        [(0, 1), (1, 1), (2, 1)],  
        [(0, 2), (1, 2), (2, 2)],  
        [(0, 0), (1, 1), (2, 2)],  
        [(0, 2), (1, 1), (2, 0)] 
    ]
    for com in winning_com:
        if all(board[pos[0]][pos[1]]==player for pos in com):
            return True
    return False
    
def placing_X(board, row, col):
    if board[row][col] == "_":
        board[row][col] = "X" 
        return True
    else:
        print("Invalid move.")
        return False

def placing_O(board):
    available_moves = [(i, j) for i in range(3) for j in range(3) if board[i][j] == "_"]
    if available_moves:
        row, col = random.choice(available_moves)
        board[row][col] = "O"
            
def main():
    board=create_board()
    count=0

    while True:
        print()
        print_board(board)
        print()
        

        if count%2==0:
            user=input("Enter where you want to place , ")
            try:
                rows,cols=map(int,user.split(","))
                if rows in [0,1,2] and cols in [0,1,2]:
                    if placing_X(board,rows,cols):
                        count+=1
                    else:
                        continue
                else:
                    print("Invalid enter correct row and column")
                    continue
            except ValueError:
                print("Invalid input enter by commas")
                continue
        else:
            placing_O(board)
            count+=1

      
        if check_win(board,"X"):
            print_board(board)
            print("Congarts! User wins!")
            break
        elif check_win(board,"O"):
            print_board(board)
            print("AI wins")
            break
        elif count==9:
            print_board(board)
            print("It is draw")
            break
            

                

if __name__=="__main__":
    main()
