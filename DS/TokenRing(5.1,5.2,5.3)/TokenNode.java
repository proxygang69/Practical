import java.io.*;
import java.net.*;

public class TokenNode {

    static boolean hasToken = false;
    static int port;
    static String nextHost;
    static int nextPort;

    public static void main(String[] args) throws Exception {

        port = Integer.parseInt(args[0]);
        nextHost = args[1];
        nextPort = Integer.parseInt(args[2]);

        ServerSocket server = new ServerSocket(port);

        // If first node, create token
        if (args.length == 4 && args[3].equals("token")) {
            hasToken = true;
        }

        while (true) {

            if (hasToken) {
                System.out.println("Token received. Enter CS? (y/n)");
                BufferedReader br = new BufferedReader(
                        new InputStreamReader(System.in));

                String choice = br.readLine();

                if (choice.equalsIgnoreCase("y")) {
                    System.out.println("Entering Critical Section...");
                    Thread.sleep(2000);
                    System.out.println("Exiting Critical Section...");
                }

                // Send token to next
                Socket s = new Socket(nextHost, nextPort);
                DataOutputStream out = new DataOutputStream(s.getOutputStream());
                out.writeUTF("TOKEN");
                s.close();

                hasToken = false;
            }

            // Wait to receive token
            Socket incoming = server.accept();
            DataInputStream in = new DataInputStream(incoming.getInputStream());

            String msg = in.readUTF();

            if (msg.equals("TOKEN")) {
                hasToken = true;
                System.out.println("Token arrived!");
            }

            incoming.close();
        }
    }
}