
import java.rmi.*;

public class Server {
    public static void main(String[] args) {
        try {
            OperationImpl obj = new OperationImpl();
            Naming.rebind("rmi://localhost/AdditionService", obj);
            System.out.println("Addition Server Ready...");
        } catch (Exception e) {
            System.out.println(e);
        }
    }
}