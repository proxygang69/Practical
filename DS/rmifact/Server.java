
import java.rmi.*;

public class Server {
    public static void main(String[] args) {
        try {
            FactImpl obj = new FactImpl();
            Naming.rebind("rmi://localhost/FactService", obj);
            System.out.println("Factorial Server Ready...");
        } catch (Exception e) {
            System.out.println(e);
        }
    }
}