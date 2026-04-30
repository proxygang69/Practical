
import java.rmi.*;
import java.rmi.server.*;

public class FactImpl extends UnicastRemoteObject implements Fact {

    public FactImpl() throws RemoteException {
        super();
    }

    public int factorial(int n) throws RemoteException {
        int f = 1;
        for (int i = 1; i <= n; i++) {
            f *= i;
        }
        return f;
    }
}