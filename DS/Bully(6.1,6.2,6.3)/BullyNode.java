import java.io.*;
import java.net.*;
import java.util.*;

public class BullyNode {

    static int id;
    static int port;
    static Map<Integer, Integer> processPorts = new HashMap<>();
    static boolean isLeader = false;

    public static void main(String[] args) throws Exception {

        id = Integer.parseInt(args[0]);
        port = Integer.parseInt(args[1]);

        // Define other processes (ID → Port)
        processPorts.put(1, 5001);
        processPorts.put(2, 5002);
        processPorts.put(3, 5003);

        ServerSocket server = new ServerSocket(port);

        // Start listener thread
        new Thread(() -> listen(server)).start();

        // Give time for all nodes to start
        Thread.sleep(3000);

        // Start election manually
        startElection();
    }

    static void startElection() {
        System.out.println("Process " + id + " starts election");

        boolean higherAlive = false;

        for (int pid : processPorts.keySet()) {
            if (pid > id) {
                try {
                    Socket s = new Socket("localhost", processPorts.get(pid));
                    DataOutputStream out = new DataOutputStream(s.getOutputStream());

                    System.out.println("P" + id + " → P" + pid + " : ELECTION");
                    out.writeUTF("ELECTION " + id);

                    s.close();
                    higherAlive = true;

                } catch (Exception e) {
                    // process not alive
                }
            }
        }

        if (!higherAlive) {
            becomeLeader();
        }
    }

    static void becomeLeader() {
        isLeader = true;
        System.out.println("Process " + id + " becomes LEADER");

        for (int pid : processPorts.keySet()) {
            if (pid != id) {
                try {
                    Socket s = new Socket("localhost", processPorts.get(pid));
                    DataOutputStream out = new DataOutputStream(s.getOutputStream());

                    System.out.println("P" + id + " → P" + pid + " : COORDINATOR");
                    out.writeUTF("COORDINATOR " + id);

                    s.close();
                } catch (Exception e) {}
            }
        }
    }

    static void listen(ServerSocket server) {
        while (true) {
            try {
                Socket s = server.accept();
                DataInputStream in = new DataInputStream(s.getInputStream());

                String msg = in.readUTF();
                String[] parts = msg.split(" ");
                String type = parts[0];
                int sender = Integer.parseInt(parts[1]);

                if (type.equals("ELECTION")) {
                    System.out.println("P" + id + " received ELECTION from P" + sender);

                    // Send OK
                    Socket reply = new Socket("localhost", processPorts.get(sender));
                    DataOutputStream out = new DataOutputStream(reply.getOutputStream());

                    System.out.println("P" + id + " → P" + sender + " : OK");
                    out.writeUTF("OK " + id);
                    reply.close();

                    // Start own election
                    startElection();
                }

                if (type.equals("OK")) {
                    System.out.println("P" + id + " received OK from P" + sender);
                }

                if (type.equals("COORDINATOR")) {
                    System.out.println("P" + id + " acknowledges Leader = P" + sender);
                    isLeader = false;
                }

                s.close();

            } catch (Exception e) {}
        }
    }
}