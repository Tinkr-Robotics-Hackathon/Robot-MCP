# Constants
SQUARE_SIZE_MM = 30  # Each square is 3mm x 3mm

# Define gripper width per piece type (approximate)
PIECE_GRIP_WIDTH = {
    'P': 8,   # Pawn
    'N': 10,  # Knight
    'B': 10,  # Bishop
    'R': 12,  # Rook
    'Q': 14,  # Queen
    'K': 15   # King
}

def square_to_mm(square):
    """Convert board square like 'a1' to real-world X/Y in mm."""
    file = ord(square[0].lower()) - ord('a')  # a=0, h=7
    print(f"File: {file}")
    rank = int(square[1]) - 1                 # 1=0, 8=7
    x = file * SQUARE_SIZE_MM
    y = rank * SQUARE_SIZE_MM
    print(f"X, Y: {x, y}") 
    return x, y

def parse_piece_type(piece_name):
    """Extracts the type letter (P, N, etc.) from a full piece name like 'B_P'."""
    return piece_name.split("_")[1].upper()  # 'B_P' -> 'P'

def generate_robot_action_sequence(command_dict):
  """ The script that controls the robot expects commands in this format:

  move_params = {
            "move_gripper_up_mm": "10", # Will move up 1 cm
            "move_gripper_forward_mm": "-5", # Will move backward 5 mm
            "tilt_gripper_down_angle": "10", # Will tilt gripper down 10 degrees
            "rotate_gripper_clockwise_angle": "-15", # Will rotate gripper counterclockwise 15 degrees
            "rotate_robot_clockwise_angle": "15" # Will rotate robot clockwise (to the right) 15 degrees
        }

  """
  
    piece_name = command_dict["piece_name"]
    from_square = command_dict["from"]
    to_square = command_dict["to"]
    
    piece_type = parse_piece_type(piece_name)
    grip_width = PIECE_GRIP_WIDTH.get(piece_type, 10)

    from_x, from_y = square_to_mm(from_square)
    to_x, to_y = square_to_mm(to_square)

    actions = [
        {"type": "move", "x": from_x, "y": from_y, "z": 10},  # Hover above source
        {"type": "move", "z": 0},                             # Lower to grip
        {"type": "grip", "action": "close", "width": grip_width},  # Grip piece
        {"type": "move", "z": 10},                            # Lift piece
        {"type": "move", "x": to_x, "y": to_y, "z": 10},      # Hover above target
        {"type": "move", "z": 0},                             # Lower to drop
        {"type": "grip", "action": "open"},                   # Release piece
        {"type": "move", "z": 10}                             # Move up again
    ]

    return {
        "piece": piece_name,
        "from": from_square,
        "to": to_square,
        "actions": actions
    }

# Example usage
if __name__ == "__main__":
    input_dict = {
        "piece_name": "B_P",
        "from": "c3",
        "to": "e4"
    }

    from pprint import pprint
    result = generate_robot_action_sequence(input_dict)
    #pprint(result)
