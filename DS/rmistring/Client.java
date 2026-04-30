
import java.rmi.*;

public class Client {
    public static void main(String[] args) {
        try {
            StringOp obj = (StringOp) Naming.lookup("rmi://localhost/StringService");
            System.out.println("Largest String: " + obj.compare("Apple", "Banana"));
        } catch (Exception e) {
            System.out.println(e);
        }
    }
}