#include <stdio.h>
#include <stdbool.h>

#define MAX_PAGES 50       // Maximum number of pages that can be referenced
#define MAX_FRAMES 10      // Maximum number of frames that can be used

// Function to simulate Least Recently Used (LRU) page replacement
void lru(int pages[], int n, int frame_size) {
    int frames[MAX_FRAMES];             // Array to store current frames
    int last_used[MAX_FRAMES];          // Array to track the last usage time of each frame
    int page_faults = 0;                // Counter for page faults

    // Initialize frames and last_used arrays
    for (int i = 0; i < frame_size; i++) {
        frames[i] = -1;
        last_used[i] = -1;
    }

    // Iterate through each page reference
    for (int i = 0; i < n; i++) {
        int page = pages[i];            // Current page being referenced
        bool found = false;             // Flag to check if page is in frames
        int lru_index = -1;             // Index of the least recently used frame

        // Check if the current page is already in one of the frames
        for (int j = 0; j < frame_size; j++) {
            if (frames[j] == page) {    // If page is found in frames
                found = true;           // Set found flag
                last_used[j] = i;       // Update last used time
                break;                  // Exit the loop early
            }
        }

        // If the page was not found, it's a page fault
        if (!found) {
            // Find an empty frame if available, otherwise find the LRU page to replace
            lru_index = -1;
            for (int j = 0; j < frame_size; j++) {
                if (frames[j] == -1) { // Check for an empty frame
                    lru_index = j;
                    break;
                }
                if (lru_index == -1 || last_used[j] < last_used[lru_index]) {
                    lru_index = j;      // Find the least recently used frame
                }
            }
            // Replace the LRU page or fill an empty slot with the new page
            frames[lru_index] = page;
            last_used[lru_index] = i;    // Update last used time for the replaced page
            page_faults++;               // Increment the page fault counter
        }

        // Print current reference and state of frames
        printf("Reference: %d | Frames: ", page);
        for (int k = 0; k < frame_size; k++) {
            if (frames[k] != -1) {       // Only print frames that are occupied
                printf("%d ", frames[k]);
            } else {
                printf("- ");
            }
        }
        printf("\n");                    // New line for better readability
    }

    // Print total number of page faults after processing all references
    printf("Total Page Faults: %d\n", page_faults);
}

int main() {
    int pages[MAX_PAGES], n, frame_size;

    // Prompt user for number of frames and validate input
    printf("Enter number of frames (minimum 3): ");
    scanf("%d", &frame_size);
    if (frame_size < 3) {              // Ensure frame size is at least 3
        printf("Frame size must be at least 3.\n");
        return 1;                      // Exit program if invalid
    }

    // Prompt user for number of pages
    printf("Enter number of pages: ");
    scanf("%d", &n);

    // Prompt user for the page reference string
    printf("Enter the page reference string (space-separated): ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &pages[i]);        // Read each page reference
    }

    // Call the LRU function to process the page references
    lru(pages, n, frame_size);
    return 0;                          // Exit program successfully
}