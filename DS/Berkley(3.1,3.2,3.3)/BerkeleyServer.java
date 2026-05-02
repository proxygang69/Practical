import java.io.*;
import java.net.*;
import java.util.*;

public class BerkeleyServer {

    static List<Socket> clients = new ArrayList<>();
    static List<Long> clientTimes = new ArrayList<>();

    public static void main(String[] args) throws Exception {

        ServerSocket serverSocket = new ServerSocket(5000);
        System.out.println("Master waiting for clients...");

        // Accept 2 clients (you can increase)
        for (int i = 0; i < 2; i++) {
            Socket socket = serverSocket.accept();
            clients.add(socket);
            System.out.println("Client connected");
        }

        // Send request & receive time
        for (Socket s : clients) {
            DataOutputStream out = new DataOutputStream(s.getOutputStream());
            DataInputStream in = new DataInputStream(s.getInputStream());

            out.writeUTF("SEND_TIME");
            long time = in.readLong();
            clientTimes.add(time);
        }

        // Master's own time
        long masterTime = System.currentTimeMillis() / 1000;

        long sum = masterTime;
        for (long t : clientTimes) sum += t;

        long avg = sum / (clientTimes.size() + 1);

        System.out.println("Average Time: " + avg);

        // Send adjustments
        for (int i = 0; i < clients.size(); i++) {
            Socket s = clients.get(i);
            DataOutputStream out = new DataOutputStream(s.getOutputStream());

            long offset = avg - clientTimes.get(i);
            out.writeLong(offset);
        }

        long masterOffset = avg - masterTime;
        System.out.println("Master adjusts by: " + masterOffset);

        serverSocket.close();
    }
}