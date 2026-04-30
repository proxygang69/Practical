
import java.rmi.*;

public class Server {
    public static void main(String[] args) {
        try {
            StringOpImpl obj = new StringOpImpl();
            Naming.rebind("rmi://localhost/StringService", obj);
            System.out.println("String Server Ready...");
        } catch (Exception e) {
            System.out.println(e);
        }
    }
}