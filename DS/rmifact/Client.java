
import java.rmi.*;

public class Client{
    public static void main(String[] args) {
        try {
            Fact obj = (Fact) Naming.lookup("rmi://localhost/FactService");
            System.out.println("Factorial: " + obj.factorial(5));
        } catch (Exception e) {
            System.out.println(e);
        }
    }
}