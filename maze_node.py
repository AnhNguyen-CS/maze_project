import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Int32, Bool, String

from .Board import Board
from .Player import Player
from .QLearner import QLearner
from .Window import Window


class MazeNode(Node):
    def __init__(self):
        super().__init__("maze_qlearning_node")

        # ----- your existing objects -----
        self.board = Board()
        self.player = Player()
        self.qlearner = QLearner(self.board)
        self.window = Window()

        self.board.createPenaltyCells()
        self.window.drawSurface(self.board, self.player)

        # ----- ROS publishers -----
        self.state_pub = self.create_publisher(Int32MultiArray, "maze/state", 10)
        self.reward_pub = self.create_publisher(Int32, "maze/reward", 10)
        self.done_pub = self.create_publisher(Bool, "maze/done", 10)
        self.action_pub = self.create_publisher(String, "maze/action", 10)

        # run step() every 0.1s
        self.timer = self.create_timer(0.1, self.step)

        self.episode_done = False

    def choose_action(self, state):
        """
        Use the QLearner epsilon-greedy policy.
        """
        # QLearner already has epsilonGreedy(state)
        return self.qlearner.epsilonGreedy(state)

    def step(self):
        if self.episode_done:
            # you could reset here if you want continuous episodes
            return

        # current state
        state = self.player.getCurrCoords()

        # choose action
        action = self.choose_action(state)

        # move player
        self.player.move(action)
        new_state = self.player.getCurrCoords()

        # reward and Q update
        reward = self.board.getCellValue(new_state)
        self.qlearner.evalQFunction(state, action)

        # check terminal
        done = self.board.isTerminalCell(new_state)
        self.episode_done = done

        # update pygame window
        self.window.updateSurface(self.player)

        # ---- publish ROS messages ----
        state_msg = Int32MultiArray()
        state_msg.data = [new_state[0], new_state[1]]
        self.state_pub.publish(state_msg)

        self.reward_pub.publish(Int32(data=int(reward)))
        self.done_pub.publish(Bool(data=done))
        self.action_pub.publish(String(data=str(action)))

        self.get_logger().info(
            f"state={new_state}, action={action}, reward={reward}, done={done}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = MazeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
