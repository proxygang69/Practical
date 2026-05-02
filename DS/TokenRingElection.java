import java.util.*;

public class TokenRingElection {

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of processes: ");
        int n = sc.nextInt();

        int[] processes = new int[n];

        for (int i = 0; i < n; i++) {
            processes[i] = i;
        }

        System.out.print("Enter initiator process: ");
        int init = sc.nextInt();

        int current = init;
        int max = processes[init];

        System.out.println("Election started...");

        do {
            int next = (current + 1) % n;

            System.out.println("Process " + current + " sends message to " + next);

            if (processes[next] > max) {
                max = processes[next];
            }

            current = next;

        } while (current != init);

        System.out.println("Leader elected is process: " + max);
    }
}