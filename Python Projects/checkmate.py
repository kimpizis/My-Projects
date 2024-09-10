class ChessPiece:
    def __init__(self, color):
        self.color = color

    def __repr__(self):
        return f'{self.symbol}{self.color[0]}'

class King(ChessPiece):
    symbol = 'K'

class Queen(ChessPiece):
    symbol = 'Q'

class Rook(ChessPiece):
    symbol = 'R'

class Bishop(ChessPiece):
    symbol = 'B'

class Knight(ChessPiece):
    symbol = 'N'

class Pawn(ChessPiece):
    symbol = 'P'

class ChessBoard:
    def __init__(self):
        self.board = self.initialize_board()

    def initialize_board(self):
        # Creating an 8x8 board
        board = [[None for _ in range(8)] for _ in range(8)]
        
        # Place black pieces
        board[0] = [
            Rook('black'), Knight('black'), Bishop('black'), Queen('black'),
            King('black'), Bishop('black'), Knight('black'), Rook('black')
        ]
        board[1] = [Pawn('black') for _ in range(8)]
        
        # Place white pieces
        board[6] = [Pawn('white') for _ in range(8)]
        board[7] = [
            Rook('white'), Knight('white'), Bishop('white'), Queen('white'),
            King('white'), Bishop('white'), Knight('white'), Rook('white')
        ]
        
        return board

    def print_board(self):
        print("  a  b  c  d  e  f  g  h")
        print(" +------------------------+")
        for i, row in enumerate(self.board):
            row_repr = f"{8-i}|"  # Row number (8 to 1)
            for piece in row:
                if piece is None:
                    row_repr += " . "
                else:
                    row_repr += f" {piece} "
            print(row_repr + f"|{8-i}")
        print(" +------------------------+")
        print("  a  b  c  d  e  f  g  h")

    def is_valid_move(self, start, end):
        # For now, just check if the destination is empty or has an opponent's piece
        start_row, start_col = start
        end_row, end_col = end

        piece = self.board[start_row][start_col]
        if piece is None:
            return False
        
        destination = self.board[end_row][end_col]
        if destination is None or destination.color != piece.color:
            return True

        return False

    def move_piece(self, start, end):
        start_row, start_col = start
        end_row, end_col = end
        
        if self.is_valid_move(start, end):
            # Move the piece
            self.board[end_row][end_col] = self.board[start_row][start_col]
            self.board[start_row][start_col] = None
        else:
            print("Invalid move.")

    def get_position(self, notation):
        col, row = notation
        col = ord(col) - ord('a')
        row = 8 - int(row)
        return row, col

def play_game():
    board = ChessBoard()
    current_turn = 'white'

    while True:
        board.print_board()
        print(f"{current_turn.capitalize()}'s turn.")

        move = input("Enter your move (e.g., e2 e4): ").lower()
        if len(move.split()) != 2:
            print("Invalid input. Please enter the move in the correct format.")
            continue

        start, end = move.split()
        start_pos = board.get_position(start)
        end_pos = board.get_position(end)

        piece = board.board[start_pos[0]][start_pos[1]]
        if piece is None or piece.color != current_turn:
            print("Invalid move. You must move your own piece.")
            continue

        board.move_piece(start_pos, end_pos)

        # Switch turns
        current_turn = 'black' if current_turn == 'white' else 'white'

if __name__ == "__main__":
    play_game()
