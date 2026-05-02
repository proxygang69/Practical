import ReverseModule.*;
import org.omg.CORBA.*;
import org.omg.CosNaming.*;

public class ReverseClient {
    public static void main(String args[]) {
        try {
            ORB orb = ORB.init(args, null);

            org.omg.CORBA.Object objRef =
                orb.resolve_initial_references("NameService");

            NamingContextExt ncRef =
                NamingContextExtHelper.narrow(objRef);

            Reverse reverse = ReverseHelper.narrow(
                ncRef.resolve_str("Reverse")
            );

            String result = reverse.reverseString("Hello CORBA");
            System.out.println("Reversed: " + result);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}