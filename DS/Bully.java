import java.util.Scanner;

public class Bully {

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of processes: ");
        int n = sc.nextInt();

        boolean[] active = new boolean[n];

        for (int i = 0; i < n; i++) {
            active[i] = true;
        }

        System.out.print("Enter failed process: ");
        int fail = sc.nextInt();
        active[fail] = false;

        System.out.print("Enter initiator: ");
        int init = sc.nextInt();

        int leader = init;

        for (int i = init + 1; i < n; i++) {
            if (active[i]) {
                System.out.println("Process " + init + " sends election to " + i);
                leader = i;
            }
        }

        System.out.println("Leader is process: " + leader);
    }
}