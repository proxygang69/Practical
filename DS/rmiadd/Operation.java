
import java.rmi.*;

public interface Operation extends Remote {
    double add(double a, double b) throws RemoteException;
}