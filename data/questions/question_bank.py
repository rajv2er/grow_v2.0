"""A deterministic 200-question bank: MCQs at three levels plus one subjective item per concept.

Each topic has Easy, Medium, and Hard MCQs and a subjective explanation question,
every one carrying a numeric difficulty_rating in [0.1, 1.0]. This is essential to
the adaptive policy: every recommended target band can be fulfilled by a real item
instead of silently falling back to another difficulty.
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
    labels = ["A", "B", "C", "D"]
    level_ratings = {"Easy": 0.25, "Medium": 0.55, "Hard": 0.85}
    for subject_index, (subject, facts) in enumerate(TOPIC_FACTS.items(), start=1):
        for topic_index, (topic, fact, distractors, base_difficulty) in enumerate(facts, start=1):
            prefix = f"Q{subject_index:02d}{topic_index:02d}"
            prompts = [
                ("A", "Easy", f"Which statement best describes {topic}?"),
                ("B", "Medium", f"In a practical problem involving {topic}, which principle should guide your solution?"),
                ("C", "Hard", f"Which design decision is most consistent with the core principle of {topic}?"),
            ]
            for difficulty_index, (suffix, difficulty, prompt) in enumerate(prompts):
                correct_index = (subject_index + topic_index + difficulty_index) % len(labels)
                options = list(distractors)
                options.insert(correct_index, fact)
                questions.append({
                    "question_id": f"{prefix}{suffix}", "subject": subject, "topic": topic,
                    "question": prompt, "question_type": "MCQ", "difficulty": difficulty,
                    "difficulty_rating": _rating(level_ratings[difficulty], subject_index, topic_index, difficulty_index),
                    "option_a": options[0], "option_b": options[1], "option_c": options[2], "option_d": options[3],
                    "correct_answer": labels[correct_index], "model_answer": None,
                    "explanation": fact.capitalize() + ".",
                })
            questions.append({
                "question_id": f"{prefix}S", "subject": subject, "topic": topic,
                "question": f"Explain the core principle of {topic} in your own words, and describe one situation where you would apply it.",
                "question_type": "Subjective", "difficulty": base_difficulty,
                "difficulty_rating": _rating(level_ratings[base_difficulty], subject_index, topic_index, 3),
                "option_a": None, "option_b": None, "option_c": None, "option_d": None,
                "correct_answer": None, "model_answer": fact.capitalize() + ".",
                "explanation": f"Model answer: {fact.capitalize()}.",
            })
    # 40 new questions across 4 new types: TrueFalse, MultipleSelect,
    # FillInBlank, Numerical. 10 per type, distributed across the existing
    # topics so each topic has a new-style question to surface.
    questions.extend(_build_extended_types(TOPIC_FACTS))
    assert len(questions) == 300, f"Expected 200 base + 100 extended = 300, got {len(questions)}."
    return questions


def _build_extended_types(facts: dict) -> list[dict]:
    """Author 20 questions per new type: TrueFalse, MultipleSelect, FillInBlank, Numerical.

    Two rounds of 10 per type, with the second round covering the topics
    NOT touched by the first round, so every topic in the bank gets at
    least one non-MCQ question of each archetype where it makes sense.
    """
    out: list[dict] = []
    flat = [(s, t, fact, base)
            for s, items in facts.items() for (t, fact, _dist, base) in items]
    rating = {"Easy": 0.25, "Medium": 0.55, "Hard": 0.85}

    def _pick(n: int, skip: set) -> list:
        chosen = []
        for s, t, fact, base in flat:
            if (s, t) in skip or (s, t) in {(x[0], x[1]) for x in chosen}:
                continue
            chosen.append((s, t, fact, base))
            if len(chosen) >= n:
                break
        if len(chosen) < n:
            for s, t, fact, base in flat:
                if (s, t) in {(x[0], x[1]) for x in chosen}:
                    continue
                chosen.append((s, t, fact, base))
                if len(chosen) >= n:
                    break
        return chosen[:n]

    # === ROUND 1: original 10 per type, mirror of the prior patch ===
    chosen1: list = []
    for subj in ("DSA", "DBMS", "Operating Systems", "Computer Networks", "Software Engineering"):
        for s, t, fact, base in flat:
            if s == subj and (s, t) not in {(x[0], x[1]) for x in chosen1}:
                chosen1.append((s, t, fact, base))
                break
    if len(chosen1) < 10:
        for s, t, fact, base in flat:
            if (s, t) not in {(x[0], x[1]) for x in chosen1}:
                chosen1.append((s, t, fact, base))
            if len(chosen1) >= 10:
                break
    chosen1 = chosen1[:10]

    # 10 True/False (round 1)
    for i, (subj, topic, fact, base) in enumerate(chosen1, start=1):
        out.append({
            "question_id": f"QTF{i:02d}", "subject": subj, "topic": topic,
            "question": f"True or False: {fact[0].upper() + fact[1:]}.",
            "question_type": "TrueFalse", "difficulty": base,
            "difficulty_rating": rating[base],
            "option_a": "True", "option_b": "False", "option_c": None, "option_d": None,
            "correct_answer": "A", "model_answer": None,
            "explanation": f"Correct. {fact.capitalize()}.",
        })

    multi_data_1 = [
        ("DSA", "Arrays", ["elements are stored at contiguous addresses", "lookup by index is O(n)", "size can change dynamically", "memory is allocated as a single block"], ["A", "D"]),
        ("DSA", "Linked Lists", ["each node holds data and a link", "nodes are at arbitrary addresses", "indexing is O(1)", "deletion at head is O(1)"], ["A", "B", "D"]),
        ("DBMS", "SQL", ["SELECT reads rows", "UPDATE modifies rows", "DROP creates a backup", "INSERT adds new rows"], ["A", "B", "D"]),
        ("DBMS", "Transactions", ["atomicity is part of ACID", "consistency is optional", "durability persists committed data", "isolation controls concurrent views"], ["A", "C", "D"]),
        ("Operating Systems", "Processes", ["a process has its own address space", "threads of one process share memory", "two processes always share all state", "a process has a PCB"], ["A", "B", "D"]),
        ("Operating Systems", "Deadlocks", ["circular wait is one of four conditions", "mutual exclusion is required", "preemption always prevents deadlock", "hold-and-wait is one condition"], ["A", "B", "D"]),
        ("Computer Networks", "TCP", ["TCP is connection-oriented", "TCP guarantees ordered delivery", "TCP is unreliable", "TCP uses a three-way handshake"], ["A", "B", "D"]),
        ("Computer Networks", "UDP", ["UDP is connectionless", "UDP has low overhead", "UDP guarantees in-order delivery", "UDP is best for streaming"], ["A", "B", "D"]),
        ("Software Engineering", "Testing", ["unit tests target small components", "integration tests cover modules together", "a test plan is a build script", "regression tests catch new breakage"], ["A", "B", "D"]),
        ("Software Engineering", "Agile", ["iterative delivery", "frequent customer feedback", "no planning phase at all", "small cross-functional teams"], ["A", "B", "D"]),
    ]
    for i, (subj, topic, options, correct) in enumerate(multi_data_1, start=1):
        out.append({
            "question_id": f"QMS{i:02d}", "subject": subj, "topic": topic,
            "question": f"Select ALL statements that are true about {topic}.",
            "question_type": "MultipleSelect", "difficulty": "Medium",
            "difficulty_rating": rating["Medium"],
            "option_a": options[0], "option_b": options[1], "option_c": options[2], "option_d": options[3],
            "correct_answer": None, "model_answer": None,
            "explanation": f"Correct answers: {', '.join(correct)}. " + " ".join(options[ord(c) - ord('A')] for c in correct) + ".",
            "correct_answers_json": json.dumps(correct),
        })

    fib_data_1 = [
        ("DSA", "Recursion", "Every recursive function must have a ___ case to terminate.", "base"),
        ("DSA", "Trees", "In a binary search tree, the left subtree contains keys ___ than the root.", "smaller"),
        ("DBMS", "Normalization", "The goal of normalization is to reduce data ___ and update anomalies.", "redundancy"),
        ("DBMS", "Indexing", "A database ___ speeds up lookups on a column but can slow writes.", "index"),
        ("Operating Systems", "Paging", "Paging divides virtual memory into fixed-size ___.", "pages"),
        ("Operating Systems", "Threads", "Threads of the same process share the same address ___.", "space"),
        ("Computer Networks", "OSI", "The ___ layer of the OSI model provides end-to-end transport services.", "transport"),
        ("Computer Networks", "DNS", "DNS translates human-readable domain names into IP ___.", "addresses"),
        ("Software Engineering", "SDLC", "The ___ describes the phases used to develop and maintain software.", "sdlc"),
        ("Software Engineering", "UML", "A UML ___ diagram models classes and their relationships.", "class"),
    ]
    for i, (subj, topic, prompt, answer) in enumerate(fib_data_1, start=1):
        out.append({
            "question_id": f"QFB{i:02d}", "subject": subj, "topic": topic,
            "question": prompt,
            "question_type": "FillInBlank", "difficulty": "Easy",
            "difficulty_rating": rating["Easy"],
            "option_a": None, "option_b": None, "option_c": None, "option_d": None,
            "correct_answer": None, "model_answer": None,
            "explanation": f"The missing word is '{answer}'.",
            "blanks_json": json.dumps([answer, answer.capitalize()]),
        })

    num_data_1 = [
        ("DSA", "Arrays", "An array of 8 bytes per element starts at address 1000. What is the address of the 4th element (0-indexed)?", 1024, 0.0),
        ("DSA", "Hashing", "A hash table of size 10 stores key 47 with hash h(k)=k mod 10. What is its index?", 7, 0.0),
        ("Operating Systems", "CPU Scheduling", "Round-robin with quantum 4 ms, 3 processes of 10 ms each. What is the average waiting time (ms)?", 12, 0.05),
        ("Operating Systems", "Paging", "A page table has 8 entries; each page is 4 KB. What is the total virtual memory in KB?", 32, 0.0),
        ("DBMS", "Normalization", "A relation is in 1NF, 2NF, and 3NF. How many normal forms does it satisfy (count)?", 3, 0.0),
        ("DBMS", "Indexing", "A B+ tree of order 4 holds 64 keys. What is the minimum tree height (count of levels from root to leaf)?", 3, 0.0),
        ("Computer Networks", "IP Addressing", "An IPv4 address has how many bits in total?", 32, 0.0),
        ("Computer Networks", "OSI", "How many layers are in the OSI model?", 7, 0.0),
        ("Software Engineering", "Testing", "You write 50 unit tests; 46 pass. What percentage passed (whole number)?", 92, 0.0),
        ("Software Engineering", "SDLC", "How many phases are in the classic Waterfall model (counting requirements through maintenance)?", 6, 0.05),
    ]
    for i, (subj, topic, prompt, expected, tolerance) in enumerate(num_data_1, start=1):
        out.append({
            "question_id": f"QNU{i:02d}", "subject": subj, "topic": topic,
            "question": prompt,
            "question_type": "Numerical", "difficulty": "Medium",
            "difficulty_rating": rating["Medium"],
            "option_a": None, "option_b": None, "option_c": None, "option_d": None,
            "correct_answer": None, "model_answer": None,
            "explanation": f"Expected answer: {expected} (relative tolerance {tolerance:.0%}).",
            "expected_value": float(expected),
            "tolerance": float(tolerance),
        })

    # === ROUND 2: 10 more per type, different topics, more variety ===
    chosen2 = _pick(10, {(c[0], c[1]) for c in chosen1})

    # Round 2 True/False: negation style so they're not all the same phrasing.
    for i, (subj, topic, fact, base) in enumerate(chosen2, start=11):
        out.append({
            "question_id": f"QTF{i:02d}", "subject": subj, "topic": topic,
            "question": f"True or False: It is NOT the case that {fact[0].lower() + fact[1:]}",
            "question_type": "TrueFalse", "difficulty": base,
            "difficulty_rating": rating[base],
            "option_a": "True", "option_b": "False", "option_c": None, "option_d": None,
            "correct_answer": "B", "model_answer": None,
            "explanation": f"The original statement is true, so its negation is false. {fact.capitalize()}.",
        })

    multi_data_2 = [
        ("DSA", "Strings", ["immutable in many languages", "support concatenation with +", "stored as a 2D array of chars", "have a length property"], ["A", "B", "D"]),
        ("DSA", "Stack & Queue", ["LIFO ordering", "push and pop are O(1)", "useful for recursion", "best for random access by index"], ["A", "B", "C"]),
        ("DBMS", "Joins", ["INNER JOIN drops unmatched rows", "LEFT JOIN keeps all left rows", "FULL OUTER JOIN is not in standard SQL", "JOIN runs in the WHERE clause by default"], ["A", "B", "D"]),
        ("DBMS", "Keys", ["primary key uniquely identifies a row", "a foreign key references another table's primary key", "any column can be a primary key", "composite keys use multiple columns"], ["A", "B", "D"]),
        ("Operating Systems", "Memory Management", ["contiguous allocation supports base+limit", "paging avoids external fragmentation", "segmentation uses variable-sized units", "swapping moves processes to disk"], ["A", "B", "C", "D"]),
        ("Operating Systems", "Synchronization", ["a mutex provides mutual exclusion", "semaphores can be binary or counting", "spinlocks block the process on contention", "monitors encapsulate shared state"], ["A", "B", "D"]),
        ("Computer Networks", "HTTP", ["HTTP is a request-response protocol", "HTTPS adds TLS on top of HTTP", "HTTP/2 multiplexes streams", "HTTP is connection-oriented like TCP"], ["A", "B", "C"]),
        ("Computer Networks", "Routing", ["OSPF is a link-state protocol", "BGP is the internet's exterior gateway protocol", "distance vector uses Bellman-Ford", "static routes always beat dynamic ones"], ["A", "B", "C"]),
        ("Software Engineering", "Design Patterns", ["Singleton restricts instantiation to one", "Observer notifies dependents", "MVC separates model, view, controller", "Factory creates a family of related objects"], ["A", "B", "C", "D"]),
        ("Software Engineering", "Git/Version Control", ["commits are immutable", "branches are pointers to commits", "rebasing rewrites history", "merge always creates a fast-forward"], ["A", "B", "C"]),
    ]
    for i, (subj, topic, options, correct) in enumerate(multi_data_2, start=11):
        out.append({
            "question_id": f"QMS{i:02d}", "subject": subj, "topic": topic,
            "question": f"Which of the following apply to {topic}? (Select ALL.)",
            "question_type": "MultipleSelect", "difficulty": "Medium",
            "difficulty_rating": rating["Medium"],
            "option_a": options[0], "option_b": options[1], "option_c": options[2], "option_d": options[3],
            "correct_answer": None, "model_answer": None,
            "explanation": f"Correct answers: {', '.join(correct)}. " + " ".join(options[ord(c) - ord('A')] for c in correct) + ".",
            "correct_answers_json": json.dumps(correct),
        })

    fib_data_2 = [
        ("DSA", "Hashing", "A good hash function distributes keys ___ across the table.", "uniformly"),
        ("DSA", "Graphs", "BFS uses a ___ while DFS uses a stack.", "queue"),
        ("DBMS", "ER Model", "In an ER diagram, a ___ connects two entities and represents an interaction.", "relationship"),
        ("DBMS", "Relational Algebra", "The ___ operation combines rows from two relations based on a condition.", "join"),
        ("Operating Systems", "CPU Scheduling", "FCFS stands for First-Come, First-___.", "served"),
        ("Operating Systems", "File Systems", "An ___ is a logical block allocation unit on disk.", "inode"),
        ("Computer Networks", "DHCP", "DHCP dynamically assigns IP ___, subnet mask, and gateway.", "addresses"),
        ("Computer Networks", "Network Security", "A ___ filters traffic between networks based on rules.", "firewall"),
        ("Software Engineering", "Waterfall", "In Waterfall, each phase must be ___ before the next begins.", "completed"),
        ("Software Engineering", "Software Quality", "___ testing verifies the system meets its requirements.", "acceptance"),
    ]
    for i, (subj, topic, prompt, answer) in enumerate(fib_data_2, start=11):
        out.append({
            "question_id": f"QFB{i:02d}", "subject": subj, "topic": topic,
            "question": prompt,
            "question_type": "FillInBlank", "difficulty": "Easy",
            "difficulty_rating": rating["Easy"],
            "option_a": None, "option_b": None, "option_c": None, "option_d": None,
            "correct_answer": None, "model_answer": None,
            "explanation": f"The missing word is '{answer}'.",
            "blanks_json": json.dumps([answer, answer.capitalize()]),
        })

    num_data_2 = [
        ("DSA", "Strings", "ASCII encodes each character in 8 bits. How many distinct characters can plain ASCII represent?", 128, 0.0),
        ("DSA", "Stack & Queue", "A circular queue of capacity 8 has front=3, rear=5. After 2 enqueues the new rear index (mod 8) is?", 7, 0.0),
        ("DBMS", "Joins", "Two tables A and B have 10 and 5 rows respectively. The cross join A × B has how many rows?", 50, 0.0),
        ("DBMS", "Keys", "How many columns are in a composite key with 3 attributes?", 3, 0.0),
        ("Operating Systems", "Deadlocks", "How many of Coffman's conditions are required for a deadlock (hold-and-wait, no-preemption, circular wait, mutual exclusion)?", 4, 0.0),
        ("Operating Systems", "Memory Management", "A system has 16 KB pages and a 32-bit address space. How many page-table entries are needed?", 1048576, 0.0),
        ("Computer Networks", "HTTP", "HTTP status code 404 means?", 404, 0.0),
        ("Computer Networks", "TCP", "TCP's three-way handshake uses how many packets to establish a connection?", 3, 0.0),
        ("Software Engineering", "UML", "How many basic relationship types are there in a UML class diagram (association, inheritance, realization, dependency)?", 4, 0.0),
        ("Software Engineering", "Project Management", "A project has 100 story points; 4 engineers each work 5 days at 5 points/day. How many days to finish?", 1, 0.0),
    ]
    for i, (subj, topic, prompt, expected, tolerance) in enumerate(num_data_2, start=11):
        out.append({
            "question_id": f"QNU{i:02d}", "subject": subj, "topic": topic,
            "question": prompt,
            "question_type": "Numerical", "difficulty": "Medium",
            "difficulty_rating": rating["Medium"],
            "option_a": None, "option_b": None, "option_c": None, "option_d": None,
            "correct_answer": None, "model_answer": None,
            "explanation": f"Expected answer: {expected} (relative tolerance {tolerance:.0%}).",
            "expected_value": float(expected),
            "tolerance": float(tolerance),
        })

    # === 10 new MCQs (Hand-written, cross-subject) ===
    new_mcqs = [
        # (subject, topic, difficulty, prompt, A, B, C, D, correct_letter, explanation)
        ("DSA", "Stack & Queue", "Easy",
         "Which data structure follows Last-In-First-Out (LIFO) order?",
         "Stack", "Queue", "Hash table", "Linked list", "A",
         "Stack pushes and pops from the top, giving LIFO order."),
        ("DSA", "Trees", "Medium",
         "What is the worst-case time complexity of searching in an unbalanced binary search tree with n nodes?",
         "O(log n)", "O(n)", "O(n log n)", "O(1)", "B",
         "An unbalanced BST can degenerate into a linked list, giving O(n) search."),
        ("DSA", "Dynamic Programming", "Hard",
         "Memoisation primarily helps with which problem property?",
         "Greedy choice", "Optimal substructure with overlapping subproblems", "Random access only", "Hash collisions", "B",
         "DP applies when optimal solutions combine optimal sub-solutions and subproblems repeat."),
        ("DBMS", "Joins", "Easy",
         "Which JOIN keeps all rows from the left table and only matching rows from the right?",
         "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "CROSS JOIN", "B",
         "LEFT JOIN preserves all left rows; right-side columns are NULL when no match."),
        ("DBMS", "Normalization", "Medium",
         "Which normal form removes transitive functional dependencies?",
         "1NF", "2NF", "3NF", "BCNF", "C",
         "Third Normal Form eliminates transitive dependencies (non-key attr -> non-key attr)."),
        ("DBMS", "Indexing", "Hard",
         "A clustered index determines the ___ order of a table's data rows.",
         "logical", "physical", "lexicographic only", "alphabetical only", "B",
         "A clustered index sorts the table's actual data rows on disk by the indexed column."),
        ("Operating Systems", "Memory Management", "Easy",
         "Which allocation scheme suffers from external fragmentation?",
         "Paging", "Segmentation", "Buddy system", "Slab allocator", "B",
         "Variable-sized segments can leave external holes between allocated blocks."),
        ("Operating Systems", "File Systems", "Medium",
         "Which data structure is typically used to map file paths to on-disk inodes?",
         "Stack", "B+ tree", "Hash table", "Linked list", "B",
         "Most file systems use a B+ tree for directory entries for ordered range queries."),
        ("Operating Systems", "I/O Management", "Hard",
         "DMA stands for?",
         "Direct Memory Access", "Dual Mode Addressing", "Dynamic Memory Allocation", "Device Mode Arbitration", "A",
         "DMA lets peripherals transfer data to/from memory without CPU involvement."),
        ("Computer Networks", "Routing", "Easy",
         "Which routing protocol is path-vector based and used between autonomous systems on the internet?",
         "OSPF", "RIP", "BGP", "EIGRP", "C",
         "BGP is the de-facto inter-AS routing protocol using path-vector semantics."),
    ]
    for i, (subj, topic, diff, prompt, a, b, c, d, correct, expl) in enumerate(new_mcqs, start=1):
        out.append({
            "question_id": f"QMC2{i:02d}", "subject": subj, "topic": topic,
            "question": prompt, "question_type": "MCQ", "difficulty": diff,
            "difficulty_rating": rating[diff],
            "option_a": a, "option_b": b, "option_c": c, "option_d": d,
            "correct_answer": correct, "model_answer": None,
            "explanation": expl,
        })

    # === 10 new Subjectives (Hand-written, cross-subject) ===
    new_subjective = [
        ("DSA", "Dynamic Programming", "Hard", "Pick a classic DP problem (Knapsack, LCS, or Edit Distance) and walk through its recurrence. Explain what the subproblems represent and why the table is filled bottom-up."),
        ("DBMS", "Transactions", "Hard", "Describe the ACID properties with one concrete example each. Why is isolation difficult to achieve without serializability, and what does the database trade off?"),
        ("Operating Systems", "Deadlocks", "Hard", "Explain the four Coffman conditions. For Banker's algorithm, describe what the safety check does and why it must run on every resource request."),
        ("Computer Networks", "TCP", "Hard", "Walk through TCP's three-way handshake. Why is a three-step exchange (and not two) required to establish a connection? What does each side learn?"),
        ("Software Engineering", "Design Patterns", "Hard", "Pick one pattern (Observer, Strategy, or Decorator). Describe the problem it solves, the structure (roles and responsibilities), and a real-world example you would implement it for."),
        ("DSA", "Graphs", "Medium", "Compare BFS and DFS for: (a) finding shortest path in an unweighted graph, (b) detecting cycles, (c) topological sort. When would you choose one over the other?"),
        ("DBMS", "Indexing", "Medium", "Why does adding an index speed up reads but slow down writes? Describe the cost trade-off in terms of storage, write amplification, and query planner behaviour."),
        ("Operating Systems", "Synchronization", "Medium", "Compare mutexes, semaphores, and monitors. Give a concrete scenario where each is the right primitive and explain why."),
        ("Computer Networks", "HTTP", "Medium", "Explain the differences between HTTP/1.1, HTTP/2, and HTTP/3 at the transport and framing layer. What problem does each generation solve?"),
        ("Software Engineering", "Testing", "Medium", "Differentiate unit, integration, system, and acceptance testing. Give one example test you would write at each level for a small login API."),
    ]
    for i, (subj, topic, diff, prompt) in enumerate(new_subjective, start=1):
        out.append({
            "question_id": f"QSUB2{i:02d}S", "subject": subj, "topic": topic,
            "question": prompt, "question_type": "Subjective", "difficulty": diff,
            "difficulty_rating": rating[diff],
            "option_a": None, "option_b": None, "option_c": None, "option_d": None,
            "correct_answer": None, "model_answer": prompt,
            "explanation": f"A strong answer covers the core idea, uses a concrete example, and connects to a real-world trade-off or trade-off the system would make.",
        })

    return out


def _rating(base: float, subject_index: int, topic_index: int, variant_index: int, step: float = 0.10) -> float:
    """Deterministic spread of ±1 step around each band centre so the ±0.15 band has real items."""
    offset = ((subject_index + topic_index + variant_index) % 3 - 1) * step
    return round(min(1.0, max(0.1, base + offset)), 2)


def export_question_bank(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_question_bank(), indent=2), encoding="utf-8")
    return path
