
import java.rmi.*;

public class Client {
    public static void main(String[] args) {
        try {
            Operation obj = (Operation) Naming.lookup("rmi://localhost/AdditionService");
            System.out.println("Result: " + obj.add(10, 5));
        } catch (Exception e) {
            System.out.println(e);
        }
    }
}