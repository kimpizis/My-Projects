import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.Timer;

public class BrickBreakerGame extends JPanel implements KeyListener, ActionListener {

    // Game state variables
    private boolean play = false;  // Is the game in progress
    private int score = 0;  // Player score
    private int totalBricks = 21;  // Total bricks
    private Timer timer;
    private int delay = 8;  // Timer delay for game loop
    private int playerX = 310;  // Paddle position X
    private int ballPosX = 120;  // Ball position X
    private int ballPosY = 350;  // Ball position Y
    private int ballDirX = -1;  // Ball direction X
    private int ballDirY = -2;  // Ball direction Y

    private int[][] bricks;  // 2D array for the brick layout
    private int brickWidth;  // Width of each brick
    private int brickHeight;  // Height of each brick

    // Constructor
    public BrickBreakerGame() {
        bricks = new int[3][7];  // 3 rows, 7 columns
        for (int i = 0; i < bricks.length; i++) {
            for (int j = 0; j < bricks[0].length; j++) {
                bricks[i][j] = 1;  // Set all bricks as visible
            }
        }

        brickWidth = 540 / 7;  // Calculate brick width based on window size
        brickHeight = 150 / 3;  // Calculate brick height

        addKeyListener(this);
        setFocusable(true);
        setFocusTraversalKeysEnabled(false);
        timer = new Timer(delay, this);
        timer.start();
    }

    // Paint method to draw the game components
    public void paint(Graphics g) {
        // Background
        g.setColor(Color.black);
        g.fillRect(1, 1, 692, 592);

        // Draw bricks
        drawBricks((Graphics2D) g);

        // Paddle
        g.setColor(Color.green);
        g.fillRect(playerX, 550, 100, 8);

        // Ball
        g.setColor(Color.yellow);
        g.fillOval(ballPosX, ballPosY, 20, 20);

        // Scores
        g.setColor(Color.white);
        g.drawString("Score: " + score, 590, 30);

        // Win or Game Over messages
        if (totalBricks <= 0) {
            play = false;
            ballDirX = 0;
            ballDirY = 0;
            g.setColor(Color.red);
            g.drawString("You Won!", 260, 300);
        }

        if (ballPosY > 570) {
            play = false;
            ballDirX = 0;
            ballDirY = 0;
            g.setColor(Color.red);
            g.drawString("Game Over, Score: " + score, 190, 300);
        }

        g.dispose();
    }

    // Method to draw the bricks
    public void drawBricks(Graphics2D g) {
        for (int i = 0; i < bricks.length; i++) {
            for (int j = 0; j < bricks[0].length; j++) {
                if (bricks[i][j] > 0) {
                    g.setColor(Color.white);
                    g.fillRect(j * brickWidth + 80, i * brickHeight + 50, brickWidth, brickHeight);

                    // Draw brick borders
                    g.setStroke(new java.awt.BasicStroke(3));
                    g.setColor(Color.black);
                    g.drawRect(j * brickWidth + 80, i * brickHeight + 50, brickWidth, brickHeight);
                }
            }
        }
    }

    // ActionPerformed method to handle game loop
    @Override
    public void actionPerformed(ActionEvent e) {
        timer.start();
        if (play) {
            // Ball and paddle collision
            if (new Rectangle(ballPosX, ballPosY, 20, 20).intersects(new Rectangle(playerX, 550, 100, 8))) {
                ballDirY = -ballDirY;
            }

            // Ball movement
            ballPosX += ballDirX;
            ballPosY += ballDirY;

            // Ball-wall collision
            if (ballPosX < 0) {
                ballDirX = -ballDirX;
            }
            if (ballPosY < 0) {
                ballDirY = -ballDirY;
            }
            if (ballPosX > 670) {
                ballDirX = -ballDirX;
            }

            // Ball and brick collision
            for (int i = 0; i < bricks.length; i++) {
                for (int j = 0; j < bricks[0].length; j++) {
                    if (bricks[i][j] > 0) {
                        int brickX = j * brickWidth + 80;
                        int brickY = i * brickHeight + 50;
                        Rectangle brickRect = new Rectangle(brickX, brickY, brickWidth, brickHeight);

                        Rectangle ballRect = new Rectangle(ballPosX, ballPosY, 20, 20);

                        if (ballRect.intersects(brickRect)) {
                            bricks[i][j] = 0;  // Remove brick
                            totalBricks--;
                            score += 5;

                            // Ball-brick collision logic
                            if (ballPosX + 19 <= brickRect.x || ballPosX + 1 >= brickRect.x + brickWidth) {
                                ballDirX = -ballDirX;
                            } else {
                                ballDirY = -ballDirY;
                            }
                        }
                    }
                }
            }

            repaint();
        }
    }

    // KeyListener methods for paddle movement
    @Override
    public void keyPressed(KeyEvent e) {
        if (e.getKeyCode() == KeyEvent.VK_RIGHT) {
            if (playerX >= 600) {
                playerX = 600;
            } else {
                moveRight();
            }
        }

        if (e.getKeyCode() == KeyEvent.VK_LEFT) {
            if (playerX <= 10) {
                playerX = 10;
            } else {
                moveLeft();
            }
        }

        if (e.getKeyCode() == KeyEvent.VK_SPACE) {
            play = true;  // Start the game on space press
        }
    }

    public void moveRight() {
        play = true;
        playerX += 20;
    }

    public void moveLeft() {
        play = true;
        playerX -= 20;
    }

    @Override
    public void keyReleased(KeyEvent e) {}

    @Override
    public void keyTyped(KeyEvent e) {}

    // Main method to set up the JFrame
    public static void main(String[] args) {
        JFrame frame = new JFrame();
        BrickBreakerGame game = new BrickBreakerGame();

        // Set JFrame properties
        frame.setBounds(10, 10, 700, 600);
        frame.setTitle("Brick Breaker");
        frame.setResizable(false);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Add the game panel
        frame.add(game);

        // Make sure to call setVisible after adding the panel
        frame.setVisible(true);
    }
}
