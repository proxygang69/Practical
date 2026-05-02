import java.io.*;
import java.net.*;

public class BerkeleyClient {

    public static void main(String[] args) throws Exception {

        Socket socket = new Socket("localhost", 5000); // change to server IP

        DataInputStream in = new DataInputStream(socket.getInputStream());
        DataOutputStream out = new DataOutputStream(socket.getOutputStream());

        String msg = in.readUTF();

        if (msg.equals("SEND_TIME")) {
            long localTime = System.currentTimeMillis() / 1000;
            System.out.println("Local Time: " + localTime);

            out.writeLong(localTime);

            long offset = in.readLong();
            System.out.println("Adjustment received: " + offset);

            long newTime = localTime + offset;
            System.out.println("Updated Time: " + newTime);
        }

        socket.close();
    }
}