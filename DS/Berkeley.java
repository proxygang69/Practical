import java.util.*;

public class Berkeley {
    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of clocks: ");
        int n = sc.nextInt();

        int[] clocks = new int[n];

        System.out.println("Enter time of each clock:");
        for (int i = 0; i < n; i++) {
            clocks[i] = sc.nextInt();
        }

        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += clocks[i];
        }

        int avg = sum / n;

        System.out.println("Average Time: " + avg);

        System.out.println("Adjusted Times:");
        for (int i = 0; i < n; i++) {
            int diff = avg - clocks[i];
            System.out.println("Clock " + i + " adjusted by " + diff);
        }
    }
}