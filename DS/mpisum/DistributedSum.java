import mpi.*;
import java.util.Arrays;

public class DistributedSum {
    public static void main(String[] args) throws Exception {

        MPI.Init(args);

        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();

        int[] sendbuf = new int[size];
        int[] recvbuf = new int[1];
        int[] result = new int[1]; // for final sum at root

        // Root initializes data
        if (rank == 0) {
            for (int i = 0; i < size; i++) {
                sendbuf[i] = (i + 1) * 10; // 10,20,30,...
            }

            System.out.print("Root distributing: " + Arrays.toString(sendbuf));
            System.out.println();
        }

        // Scatter
        MPI.COMM_WORLD.Scatter(sendbuf, 0, 1, MPI.INT,
                               recvbuf, 0, 1, MPI.INT,
                               0);

        int localValue = recvbuf[0];

        System.out.println("Process " + rank +
                " received: " + localValue);

        // Reduce (sum all values at root)
        MPI.COMM_WORLD.Reduce(recvbuf, 0, result, 0, 1,
                              MPI.INT, MPI.SUM, 0);

        // Only root prints final result
        if (rank == 0) {
            System.out.println("Final Distributed Sum: " + result[0]);
        }

        MPI.Finalize();
    }
}