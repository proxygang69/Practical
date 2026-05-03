import java.util.*;

public class TokenRingElection {

    public static void main(String[] args) {

        // Process IDs (you can change these)
        int[] processes = {1, 2, 3, 4, 5};
        int n = processes.length;

        // Assume process 2 starts election (index 1)
        int initiatorIndex = 1;

        System.out.println("Processes in ring: " + Arrays.toString(processes));
        System.out.println("Process " + processes[initiatorIndex] + " starts election\n");

        List<Integer> token = new ArrayList<>();

        int i = initiatorIndex;

        // Pass token around the ring
        do {
            int currentProcess = processes[i];
            System.out.println("Process " + currentProcess + " received token");

            token.add(currentProcess);
            System.out.println("Process " + currentProcess + " adds its ID to token: " + token);

            // Pass to next process
            int next = (i + 1) % n;
            System.out.println("Process " + currentProcess + " passes token to Process " + processes[next] + "\n");

            i = next;

        } while (i != initiatorIndex);

        // Election complete
        System.out.println("Token returned to initiator: " + token);

        int leader = Collections.max(token);
        System.out.println("\nLeader elected is Process " + leader);

        // Announcement phase
        System.out.println("\n--- Leader Announcement ---");

        i = initiatorIndex;
        do {
            int currentProcess = processes[i];
            int next = (i + 1) % n;

            System.out.println("Process " + currentProcess + " informs Process " 
                               + processes[next] + " that Leader is " + leader);

            i = next;

        } while (i != initiatorIndex);

        System.out.println("\nElection completed successfully.");
    }
}