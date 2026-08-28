"""A deterministic 150-question bank with three levels for every concept.

Each topic has Easy, Medium, and Hard items. This is essential to the adaptive
policy: every recommended transition can be fulfilled by a real item instead of
silently falling back to another difficulty.
"""
from __future__ import annotations

import json
from pathlib import Path


# subject -> (topic, concise fact, misconception distractors, difficulty)
TOPIC_FACTS: dict[str, list[tuple[str, str, list[str], str]]] = {
    "DSA": [
        ("Arrays", "array elements are stored at contiguous memory locations", ["arrays always grow without reallocation", "array lookup requires scanning every item", "arrays store only integers"], "Easy"),
        ("Strings", "strings are sequences of characters", ["strings cannot be indexed", "strings are always mutable", "strings contain only letters"], "Easy"),
        ("Linked Lists", "a linked-list node stores data and a link to another node", ["nodes must occupy contiguous memory", "linked lists provide constant-time indexing", "a list cannot be empty"], "Medium"),
        ("Stack & Queue", "a stack follows Last-In, First-Out order", ["a stack follows First-In, First-Out order", "a stack removes arbitrary elements", "a stack needs sorted data"], "Easy"),
        ("Recursion", "a recursive algorithm needs a base case to terminate", ["recursion never uses a call stack", "every recursion is faster than iteration", "a base case repeats the call"], "Medium"),
        ("Trees", "a binary-search-tree left subtree contains smaller keys", ["all tree nodes have two children", "a BST is always balanced", "the right subtree contains smaller keys"], "Medium"),
        ("Graphs", "breadth-first search explores vertices level by level", ["BFS always uses a stack", "BFS only works on trees", "BFS requires weighted edges"], "Medium"),
        ("Hashing", "a hash function maps a key to an index or bucket", ["hashing sorts every key", "collisions are impossible", "a hash table uses binary search"], "Medium"),
        ("Heap", "a max-heap keeps the largest key at the root", ["a heap is fully sorted", "a min-heap has the smallest key at every leaf", "heap insertion is always O(n)"], "Medium"),
        ("Dynamic Programming", "dynamic programming reuses solutions to overlapping subproblems", ["DP ignores previous results", "DP works only with graphs", "DP requires recursion only"], "Hard"),
    ],
    "DBMS": [
        ("DBMS Basics", "a DBMS manages, stores, and retrieves structured data", ["a DBMS is only a programming language", "a DBMS cannot enforce constraints", "a DBMS stores no metadata"], "Easy"),
        ("Keys", "a primary key uniquely identifies a row", ["a primary key may contain duplicate values", "a primary key is always a foreign key", "a primary key stores only numbers"], "Easy"),
        ("SQL", "SELECT is used to retrieve data from a table", ["SELECT permanently deletes rows", "SELECT creates a transaction", "SELECT only changes schema"], "Easy"),
        ("Normalization", "normalization reduces redundancy and update anomalies", ["normalization guarantees faster every query", "normalization removes all relationships", "normalization duplicates attributes"], "Medium"),
        ("Joins", "an INNER JOIN returns rows with matching values in both tables", ["an INNER JOIN returns every unmatched row", "an INNER JOIN deletes unmatched rows", "an INNER JOIN needs no join condition"], "Medium"),
        ("Transactions", "ACID atomicity means a transaction happens completely or not at all", ["atomicity means data is encrypted", "atomicity permits partial commits", "atomicity removes concurrency"], "Medium"),
        ("Indexing", "an index can speed up record lookup for selected attributes", ["an index always reduces write cost", "an index stores no keys", "an index makes table scans mandatory"], "Medium"),
        ("ER Model", "an ER relationship represents an association between entities", ["an entity is a query result only", "a relationship cannot have cardinality", "an ER diagram contains no attributes"], "Easy"),
        ("Relational Algebra", "selection filters rows that satisfy a condition", ["selection combines two relations", "selection renames every attribute", "selection sorts a table"], "Hard"),
        ("Concurrency Control", "two-phase locking controls concurrent transaction conflicts", ["2PL prevents all database reads", "2PL makes transactions non-atomic", "2PL never uses locks"], "Hard"),
    ],
    "Operating Systems": [
        ("Processes", "a process is a program in execution", ["a process is only source code", "processes never have state", "a process cannot own resources"], "Easy"),
        ("CPU Scheduling", "round-robin scheduling uses a time quantum", ["round-robin always runs one process forever", "round-robin is non-preemptive", "round-robin has no ready queue"], "Medium"),
        ("Deadlocks", "deadlock requires circular wait among its necessary conditions", ["deadlock occurs only with one process", "deadlock means every job is complete", "deadlock cannot involve resources"], "Hard"),
        ("Memory Management", "virtual memory lets a process use an address space beyond physical RAM", ["virtual memory eliminates secondary storage", "virtual memory has no address translation", "virtual memory is only CPU cache"], "Medium"),
        ("Paging", "paging divides virtual memory into fixed-size pages", ["pages always have variable size", "paging removes page tables", "paging uses only contiguous allocation"], "Medium"),
        ("Synchronization", "a mutex provides mutual exclusion for a critical section", ["a mutex allows every thread simultaneously", "a mutex schedules disks", "a mutex replaces all locks with polling"], "Hard"),
        ("File Systems", "a file system organizes files and directories on storage", ["a file system is a CPU scheduler", "file systems cannot store metadata", "a file name is the entire file system"], "Easy"),
        ("Threads", "threads in one process share its address space", ["threads never share process resources", "each thread is a separate computer", "a thread cannot be scheduled"], "Medium"),
        ("I/O Management", "an interrupt lets a device notify the CPU about an event", ["an interrupt is a type of permanent storage", "interrupts are only user programs", "interrupts cannot be disabled"], "Medium"),
        ("Security", "least privilege grants only permissions necessary for a task", ["least privilege grants administrator access by default", "least privilege removes authentication", "least privilege shares all accounts"], "Hard"),
    ],
    "Computer Networks": [
        ("OSI", "the OSI transport layer provides end-to-end transport services", ["the transport layer assigns MAC addresses", "the transport layer is the physical cable", "the transport layer is only for DNS"], "Medium"),
        ("TCP/IP", "the Internet Protocol routes packets between networks", ["IP guarantees delivery and ordering", "IP is a physical connector", "IP encrypts every payload by itself"], "Medium"),
        ("TCP", "TCP provides reliable ordered byte-stream delivery", ["TCP is connectionless", "TCP never retransmits data", "TCP has no flow control"], "Medium"),
        ("UDP", "UDP is connectionless and has low protocol overhead", ["UDP guarantees in-order delivery", "UDP requires a three-way handshake", "UDP cannot carry application data"], "Easy"),
        ("DNS", "DNS maps domain names to IP addresses", ["DNS assigns MAC addresses", "DNS transfers web pages", "DNS is a routing protocol"], "Easy"),
        ("HTTP", "HTTP is an application-layer protocol for web communication", ["HTTP assigns IP addresses", "HTTP is a physical layer standard", "HTTP only works without servers"], "Easy"),
        ("DHCP", "DHCP dynamically assigns network configuration such as IP addresses", ["DHCP resolves domain names", "DHCP encrypts TCP streams", "DHCP is a file system"], "Medium"),
        ("IP Addressing", "a subnet mask identifies network and host portions of an IPv4 address", ["a subnet mask is a transport port", "a subnet mask is a DNS record", "a subnet mask is a MAC address"], "Medium"),
        ("Routing", "a router forwards packets using routing-table information", ["a router only switches within one Ethernet frame", "a router has no network-layer role", "a router cannot connect networks"], "Hard"),
        ("Network Security", "a firewall filters network traffic according to policy", ["a firewall replaces all authentication", "a firewall is an IP address", "a firewall stores web pages"], "Hard"),
    ],
    "Software Engineering": [
        ("SDLC", "the SDLC describes phases used to develop and maintain software", ["the SDLC is only the coding phase", "the SDLC has no testing", "the SDLC applies only after deployment"], "Easy"),
        ("Agile", "Agile development uses iterative delivery and frequent feedback", ["Agile prohibits customer feedback", "Agile requires one final delivery only", "Agile eliminates planning entirely"], "Easy"),
        ("Waterfall", "Waterfall organizes development in sequential phases", ["Waterfall has no requirements phase", "Waterfall is a network protocol", "Waterfall requires daily releases"], "Medium"),
        ("SRS", "an SRS specifies the software system requirements", ["an SRS is compiled executable code", "an SRS replaces all tests", "an SRS only lists developer salaries"], "Medium"),
        ("Testing", "unit testing verifies a small testable component in isolation", ["unit testing always tests production infrastructure", "unit tests require a complete UI", "unit testing never uses assertions"], "Medium"),
        ("UML", "a UML class diagram models classes and their relationships", ["UML only models database backups", "UML is an executable language", "UML cannot show associations"], "Medium"),
        ("Git/Version Control", "a commit records a snapshot of staged changes in Git", ["a commit permanently deletes a repository", "a commit is a network packet", "a commit cannot have a message"], "Easy"),
        ("Software Quality", "maintainability measures how easily software can be modified", ["maintainability means no documentation", "maintainability is only runtime speed", "maintainability forbids refactoring"], "Medium"),
        ("Project Management", "a work breakdown structure decomposes project work into smaller tasks", ["a WBS is a source-code compiler", "a WBS removes project scope", "a WBS is a database index"], "Medium"),
        ("Design Patterns", "the Observer pattern notifies dependent objects about state changes", ["Observer stores rows in a database", "Observer guarantees a singleton", "Observer is a network routing algorithm"], "Hard"),
    ],
}


def build_question_bank() -> list[dict]:
    questions: list[dict] = []
    for subject_index, (subject, facts) in enumerate(TOPIC_FACTS.items(), start=1):
        for topic_index, (topic, fact, distractors, base_difficulty) in enumerate(facts, start=1):
            prefix = f"Q{subject_index:02d}{topic_index:02d}"
            choices = [fact, *distractors]
            questions.append({
                "question_id": f"{prefix}A", "subject": subject, "topic": topic,
                "question": f"Which statement best describes {topic}?",
                "difficulty": "Easy",
                "option_a": choices[0], "option_b": choices[1], "option_c": choices[2], "option_d": choices[3],
                "correct_answer": "A", "explanation": fact.capitalize() + ".",
            })
            questions.append({
                "question_id": f"{prefix}B", "subject": subject, "topic": topic,
                "question": f"In a practical problem involving {topic}, which principle should guide your solution?",
                "difficulty": "Medium",
                "option_a": choices[1], "option_b": choices[2], "option_c": choices[0], "option_d": choices[3],
                "correct_answer": "C", "explanation": fact.capitalize() + ".",
            })
            questions.append({
                "question_id": f"{prefix}C", "subject": subject, "topic": topic,
                "question": f"Which design decision is most consistent with the core principle of {topic}?",
                "difficulty": "Hard",
                "option_a": choices[2], "option_b": choices[3], "option_c": choices[1], "option_d": choices[0],
                "correct_answer": "D", "explanation": fact.capitalize() + ".",
            })
    assert len(questions) == 150, "Every topic must provide Easy, Medium, and Hard practice."
    return questions


def export_question_bank(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_question_bank(), indent=2), encoding="utf-8")
    return path
