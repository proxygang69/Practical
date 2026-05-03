
import java.rmi.*;
import java.rmi.server.*;

public class OperationImpl extends UnicastRemoteObject implements Operation {

    public OperationImpl() throws RemoteException {
        super();
    }

    public double add(double a, double b) throws RemoteException {
        return a + b;
    }
}