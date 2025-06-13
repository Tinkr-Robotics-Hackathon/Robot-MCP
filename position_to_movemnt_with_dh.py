import numpy as np
import math
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional
import json

class SOLO100ChessRobot:
    def __init__(self):
        # DH Parameters for SOLO-100
        self.dh_matrix = np.array([
            [0,   np.pi/2,  330, 0],  # Joint 1 - Base rotation
            [270, 0,        0,   0],  # Joint 2 - Shoulder
            [70,  np.pi/2,  0,   0],  # Joint 3 - Elbow
            [0,   -np.pi/2, 302, 0],  # Joint 4 - Wrist pitch
            [0,   np.pi/2,  0,   0],  # Joint 5 - Wrist roll
            [0,   0,        72,  0]   # Joint 6 - End effector
        ])
        
        # Joint limits (radians)
        self.joint_limits = [
            (-np.pi, np.pi),           # Joint 1: ±180°
            (-np.pi/2, np.pi/2),       # Joint 2: ±90°
            (-np.pi, np.pi),           # Joint 3: ±180°
            (-np.pi, np.pi),           # Joint 4: ±180°
            (-np.pi/2, np.pi/2),       # Joint 5: ±90°
            (-np.pi, np.pi)            # Joint 6: ±180°
        ]
        
        # Chess board parameters
        self.SQUARE_SIZE_MM = 30
        self.BOARD_OFFSET_X = 200  # Offset from robot base to board center
        self.BOARD_OFFSET_Y = 100
        self.BOARD_HEIGHT = 5      # Height of chess board surface
        self.HOVER_HEIGHT = 15     # Height to hover above pieces
        
        # Gripper parameters
        self.PIECE_GRIP_WIDTH = {
            'P': 8,   # Pawn
            'N': 10,  # Knight
            'B': 10,  # Bishop
            'R': 12,  # Rook
            'Q': 14,  # Queen
            'K': 15   # King
        }
        
        # Current robot state
        self.current_joint_angles = [0, 0, 0, 0, 0, 0]
        self.gripper_open = True
        
    def get_dh_transform(self, a, alpha, d, theta):
        """Calculate DH transformation matrix"""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct,    -st*ca,  st*sa,   a*ct],
            [st,     ct*ca, -ct*sa,   a*st],
            [0,      sa,     ca,      d   ],
            [0,      0,      0,       1   ]
        ])
    
    def forward_kinematics(self, joint_angles):
        """Calculate end-effector pose from joint angles"""
        T_total = np.eye(4)
        
        for i, angle in enumerate(joint_angles):
            a = self.dh_matrix[i, 0]
            alpha = self.dh_matrix[i, 1]
            d = self.dh_matrix[i, 2]
            theta = angle
            
            T_i = self.get_dh_transform(a, alpha, d, theta)
            T_total = np.dot(T_total, T_i)
        
        return T_total
    
    def inverse_kinematics(self, target_position, target_orientation=None):
        """
        Solve inverse kinematics for target position and orientation
        Returns joint angles or None if no solution found
        """
        if target_orientation is None:
            # Default orientation: gripper pointing down
            target_orientation = np.array([0, 0, -1])
        
        def objective_function(angles):
            T = self.forward_kinematics(angles)
            current_pos = T[:3, 3]
            current_z_axis = T[:3, 2]
            
            pos_error = np.linalg.norm(current_pos - target_position)
            orientation_error = np.linalg.norm(current_z_axis - target_orientation)
            
            # Penalty for joint limit violations
            penalty = 0
            for i, angle in enumerate(angles):
                if angle < self.joint_limits[i][0] or angle > self.joint_limits[i][1]:
                    penalty += 1000
            
            return pos_error + 0.5 * orientation_error + penalty
        
        # Try multiple initial guesses
        best_solution = None
        best_cost = float('inf')
        
        for _ in range(10):  # Multiple random starts
            initial_guess = []
            for i in range(6):
                low, high = self.joint_limits[i]
                initial_guess.append(np.random.uniform(low, high))
            
            try:
                result = minimize(objective_function, initial_guess, method='L-BFGS-B')
                if result.success and result.fun < best_cost:
                    best_cost = result.fun
                    best_solution = result.x
            except:
                continue
        
        # Verify solution quality
        if best_solution is not None and best_cost < 5.0:  # 5mm tolerance
            return best_solution
        return None
    
    def square_to_world_coordinates(self, square):
        """Convert chess square notation to world coordinates"""
        file = ord(square[0].lower()) - ord('a')  # a=0, h=7
        rank = int(square[1]) - 1                 # 1=0, 8=7
        
        # Convert to world coordinates
        x = self.BOARD_OFFSET_X + (file - 3.5) * self.SQUARE_SIZE_MM
        y = self.BOARD_OFFSET_Y + (rank - 3.5) * self.SQUARE_SIZE_MM
        z = self.BOARD_HEIGHT
        
        return np.array([x, y, z])
    
    def parse_piece_type(self, piece_name):
        """Extract piece type from piece name"""
        return piece_name.split("_")[1].upper()
    
    def generate_trajectory(self, start_angles, end_angles, num_points=20):
        """Generate smooth trajectory between two joint configurations"""
        trajectory = []
        for i in range(num_points):
            t = i / (num_points - 1)
            # Use cubic interpolation for smooth motion
            t_smooth = 3*t**2 - 2*t**3
            
            angles = []
            for j in range(6):
                angle = start_angles[j] + t_smooth * (end_angles[j] - start_angles[j])
                angles.append(angle)
            
            trajectory.append(angles)
        
        return trajectory
    
    def move_to_position(self, target_position, target_orientation=None):
        """Move robot to target position with trajectory planning"""
        target_angles = self.inverse_kinematics(target_position, target_orientation)
        
        if target_angles is None:
            return False, "No valid solution found for target position"
        
        # Generate smooth trajectory
        trajectory = self.generate_trajectory(self.current_joint_angles, target_angles)
        
        # Execute trajectory
        robot_commands = []
        for i, angles in enumerate(trajectory):
            cmd = {
                "step": i,
                "joint_angles": [math.degrees(a) for a in angles],
                "execution_time": 0.1  # 100ms per step
            }
            robot_commands.append(cmd)
        
        # Update current state
        self.current_joint_angles = target_angles
        
        return True, robot_commands
    
    def control_gripper(self, action, width=None):
        """Control gripper open/close"""
        if action == "open":
            self.gripper_open = True
            return {"gripper_action": "open", "width": 20}
        elif action == "close":
            self.gripper_open = False
            grip_width = width if width else 10
            return {"gripper_action": "close", "width": grip_width}
        
        return None
    
    def execute_chess_move(self, command_dict):
        """
        Execute a complete chess move with the robot
        """
        piece_name = command_dict["piece_name"]
        from_square = command_dict["from"]
        to_square = command_dict["to"]
        
        # Parse piece information
        piece_type = self.parse_piece_type(piece_name)
        grip_width = self.PIECE_GRIP_WIDTH.get(piece_type, 10)
        
        # Get world coordinates
        from_pos = self.square_to_world_coordinates(from_square)
        to_pos = self.square_to_world_coordinates(to_square)
        
        # Create complete move sequence
        move_sequence = []
        
        print(f"Executing move: {piece_name} from {from_square} to {to_square}")
        print(f"From position: {from_pos}")
        print(f"To position: {to_pos}")
        
        # Step 1: Move to hover position above source square
        hover_from = from_pos.copy()
        hover_from[2] += self.HOVER_HEIGHT
        
        success, commands = self.move_to_position(hover_from)
        if not success:
            return {"success": False, "error": commands}
        
        move_sequence.extend(commands)
        
        # Step 2: Lower to pick up piece
        success, commands = self.move_to_position(from_pos)
        if not success:
            return {"success": False, "error": commands}
        
        move_sequence.extend(commands)
        
        # Step 3: Close gripper
        gripper_cmd = self.control_gripper("close", grip_width)
        move_sequence.append(gripper_cmd)
        
        # Step 4: Lift piece
        success, commands = self.move_to_position(hover_from)
        if not success:
            return {"success": False, "error": commands}
        
        move_sequence.extend(commands)
        
        # Step 5: Move to hover position above target square
        hover_to = to_pos.copy()
        hover_to[2] += self.HOVER_HEIGHT
        
        success, commands = self.move_to_position(hover_to)
        if not success:
            return {"success": False, "error": commands}
        
        move_sequence.extend(commands)
        
        # Step 6: Lower to place piece
        success, commands = self.move_to_position(to_pos)
        if not success:
            return {"success": False, "error": commands}
        
        move_sequence.extend(commands)
        
        # Step 7: Open gripper
        gripper_cmd = self.control_gripper("open")
        move_sequence.append(gripper_cmd)
        
        # Step 8: Move back to hover position
        success, commands = self.move_to_position(hover_to)
        if not success:
            return {"success": False, "error": commands}
        
        move_sequence.extend(commands)
        
        return {
            "success": True,
            "piece": piece_name,
            "from": from_square,
            "to": to_square,
            "move_sequence": move_sequence,
            "total_steps": len(move_sequence)
        }
    
    def get_robot_status(self):
        """Get current robot status"""
        current_pose = self.forward_kinematics(self.current_joint_angles)
        current_position = current_pose[:3, 3]
        
        return {
            "joint_angles_deg": [math.degrees(a) for a in self.current_joint_angles],
            "end_effector_position": current_position.tolist(),
            "gripper_open": self.gripper_open,
            "robot_ready": True
        }
    
    def calibrate_board_position(self, corner_positions):
        """
        Calibrate board position using corner square positions
        corner_positions should be dict with keys 'a1', 'a8', 'h1', 'h8'
        """
        # This would be used for real-world calibration
        # For now, we use the predefined offsets
        print("Board calibration would be performed here")
        return True

# Example usage and testing
if __name__ == "__main__":
    # Create robot controller
    robot = SOLO100ChessRobot()
    
    # Test move command
    test_move = {
        "piece_name": "W_P",  # White Pawn
        "from": "e2",
        "to": "e4"
    }
    
    print("SOLO-100 Chess Robot Controller")
    print("=" * 40)
    
    # Show initial robot status
    status = robot.get_robot_status()
    print("Initial Robot Status:")
    print(f"Joint Angles: {[f'{a:.1f}°' for a in status['joint_angles_deg']]}")
    print(f"End Effector Position: {[f'{p:.1f}mm' for p in status['end_effector_position']]}")
    print(f"Gripper Open: {status['gripper_open']}")
    print()
    
    # Execute chess move
    print("Executing chess move...")
    result = robot.execute_chess_move(test_move)
    
    if result["success"]:
        print(f"✓ Move executed successfully!")
        print(f"Piece: {result['piece']}")
        print(f"From: {result['from']} → To: {result['to']}")
        print(f"Total steps: {result['total_steps']}")
        
        # Show first few commands
        print("\nFirst 5 commands:")
        for i, cmd in enumerate(result['move_sequence'][:5]):
            print(f"  {i+1}: {cmd}")
    else:
        print(f"✗ Move failed: {result['error']}")
    
    # Show final robot status
    print("\nFinal Robot Status:")
    final_status = robot.get_robot_status()
    print(f"Joint Angles: {[f'{a:.1f}°' for a in final_status['joint_angles_deg']]}")
    print(f"End Effector Position: {[f'{p:.1f}mm' for p in final_status['end_effector_position']]}")
    print(f"Gripper Open: {final_status['gripper_open']}")
    
    # Test multiple moves
    print("\n" + "=" * 40)
    print("Testing multiple moves...")
    
    test_moves = [
        {"piece_name": "W_N", "from": "b1", "to": "c3"},
        {"piece_name": "B_P", "from": "d7", "to": "d5"},
        {"piece_name": "W_B", "from": "f1", "to": "c4"}
    ]
    
    for move in test_moves:
        result = robot.execute_chess_move(move)
        if result["success"]:
            print(f"✓ {move['piece_name']} {move['from']}→{move['to']} ({result['total_steps']} steps)")
        else:
            print(f"✗ {move['piece_name']} {move['from']}→{move['to']} FAILED")
