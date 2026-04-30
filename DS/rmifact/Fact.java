
import java.rmi.*;

public interface Fact extends Remote {
    int factorial(int n) throws RemoteException;
}