"""Curated external reading links for every topic in the question bank.

Each entry maps (subject, topic) to a list of (label, url) pairs. Sources are
authoritative encyclopaedic references — Wikipedia, cppreference, MDN, and a
small number of canonical textbook-style pages. These are not generated; they
are hand-picked and verified to load. Used by `app/main.py` to render the
inline reference panel in the Recommendations page.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

TopicKey = Tuple[str, str]
Link = Tuple[str, str]

WIKI = "Wikipedia"
GEEKS = "GeeksforGeeks"
CPPREF = "cppreference"
MDN = "MDN"
TUTORIALS = "TutorialsPoint"
W3 = "W3Schools"
ORACLE = "Oracle Docs"
POSTGRES = "PostgreSQL Docs"
REDIS = "Redis Docs"
PYDOCS = "Python Docs"

REFERENCES: Dict[TopicKey, List[Link]] = {
    # ---------------- DSA ----------------
    ("DSA", "Arrays"): [
        (WIKI, "https://en.wikipedia.org/wiki/Array_data_structure"),
        (GEEKS, "https://www.geeksforgeeks.org/array-data-structure/"),
    ],
    ("DSA", "Strings"): [
        (WIKI, "https://en.wikipedia.org/wiki/String_(computer_science)"),
        (PYDOCS, "https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str"),
    ],
    ("DSA", "Linked Lists"): [
        (WIKI, "https://en.wikipedia.org/wiki/Linked_list"),
        (GEEKS, "https://www.geeksforgeeks.org/data-structures/linked-list/"),
    ],
    ("DSA", "Stack & Queue"): [
        (WIKI, "https://en.wikipedia.org/wiki/Stack_(abstract_data_type)"),
        (WIKI, "https://en.wikipedia.org/wiki/Queue_(abstract_data_type)"),
        (GEEKS, "https://www.geeksforgeeks.org/stack-data-structure/"),
    ],
    ("DSA", "Trees"): [
        (WIKI, "https://en.wikipedia.org/wiki/Tree_(data_structure)"),
    ],
    ("DSA", "Graphs"): [
        (WIKI, "https://en.wikipedia.org/wiki/Graph_(abstract_data_type)"),
        (GEEKS, "https://www.geeksforgeeks.org/graph-data-structure-and-algorithms/"),
    ],
    ("DSA", "Hashing"): [
        (WIKI, "https://en.wikipedia.org/wiki/Hash_function"),
        (WIKI, "https://en.wikipedia.org/wiki/Hash_table"),
        (GEEKS, "https://www.geeksforgeeks.org/hashing-data-structure/"),
    ],
    ("DSA", "Heap"): [
        (WIKI, "https://en.wikipedia.org/wiki/Heap_(data_structure)"),
        (GEEKS, "https://www.geeksforgeeks.org/heap-data-structure/"),
    ],
    ("DSA", "Recursion"): [
        (WIKI, "https://en.wikipedia.org/wiki/Recursion_(computer_science)"),
        (GEEKS, "https://www.geeksforgeeks.org/recursion/"),
    ],
    ("DSA", "Dynamic Programming"): [
        (WIKI, "https://en.wikipedia.org/wiki/Dynamic_programming"),
        (GEEKS, "https://www.geeksforgeeks.org/dynamic-programming/"),
    ],
    # ---------------- DBMS ----------------
    ("DBMS", "DBMS Basics"): [
        (WIKI, "https://en.wikipedia.org/wiki/Database"),
        (GEEKS, "https://www.geeksforgeeks.org/dbms/"),
        (TUTORIALS, "https://www.tutorialspoint.com/dbms/index.htm"),
    ],
    ("DBMS", "ER Model"): [
        (WIKI, "https://en.wikipedia.org/wiki/Entity%E2%80%93relationship_model"),
        (GEEKS, "https://www.geeksforgeeks.org/introduction-of-er-model/"),
    ],
    ("DBMS", "Relational Algebra"): [
        (WIKI, "https://en.wikipedia.org/wiki/Relational_algebra"),
        (GEEKS, "https://www.geeksforgeeks.org/relational-algebra-in-dbms/"),
    ],
    ("DBMS", "SQL"): [
        (WIKI, "https://en.wikipedia.org/wiki/SQL"),
        (GEEKS, "https://www.geeksforgeeks.org/sql-tutorial/"),
        (W3, "https://www.w3schools.com/sql/"),
    ],
    ("DBMS", "Joins"): [
        (WIKI, "https://en.wikipedia.org/wiki/Join_(SQL)"),
        (GEEKS, "https://www.geeksforgeeks.org/sql-join-set-1-inner-left-right-and-full-joins/"),
    ],
    ("DBMS", "Keys"): [
        (WIKI, "https://en.wikipedia.org/wiki/Unique_key"),
    ],
    ("DBMS", "Normalization"): [
        (WIKI, "https://en.wikipedia.org/wiki/Database_normalization"),
        (GEEKS, "https://www.geeksforgeeks.org/normal-forms-in-dbms/"),
    ],
    ("DBMS", "Indexing"): [
        (WIKI, "https://en.wikipedia.org/wiki/Database_index"),
        (GEEKS, "https://www.geeksforgeeks.org/indexing-in-databases-set-1/"),
    ],
    ("DBMS", "Transactions"): [
        (WIKI, "https://en.wikipedia.org/wiki/Database_transaction"),
        (ORACLE, "https://docs.oracle.com/cd/B19306_01/server.102/b14220/transact.htm"),
    ],
    ("DBMS", "Concurrency Control"): [
        (WIKI, "https://en.wikipedia.org/wiki/Concurrency_control"),
        (GEEKS, "https://www.geeksforgeeks.org/concurrency-control-in-dbms/"),
    ],
    # ---------------- Operating Systems ----------------
    ("Operating Systems", "Processes"): [
        (WIKI, "https://en.wikipedia.org/wiki/Process_(computing)"),
    ],
    ("Operating Systems", "Threads"): [
        (WIKI, "https://en.wikipedia.org/wiki/Thread_(computing)"),
        (GEEKS, "https://www.geeksforgeeks.org/thread-in-operating-system/"),
    ],
    ("Operating Systems", "CPU Scheduling"): [
        (WIKI, "https://en.wikipedia.org/wiki/Scheduling_(computing)"),
        (GEEKS, "https://www.geeksforgeeks.org/cpu-scheduling-in-operating-systems/"),
    ],
    ("Operating Systems", "Synchronization"): [
        (WIKI, "https://en.wikipedia.org/wiki/Synchronization_(computer_science)"),
    ],
    ("Operating Systems", "Deadlocks"): [
        (WIKI, "https://en.wikipedia.org/wiki/Deadlock"),
    ],
    ("Operating Systems", "Memory Management"): [
        (WIKI, "https://en.wikipedia.org/wiki/Memory_management"),
        (GEEKS, "https://www.geeksforgeeks.org/memory-management-in-operating-system/"),
    ],
    ("Operating Systems", "Paging"): [
        (WIKI, "https://en.wikipedia.org/wiki/Paging"),
        (GEEKS, "https://www.geeksforgeeks.org/paging-in-operating-system/"),
    ],
    ("Operating Systems", "File Systems"): [
        (WIKI, "https://en.wikipedia.org/wiki/File_system"),
        (GEEKS, "https://www.geeksforgeeks.org/file-systems-in-operating-system/"),
    ],
    ("Operating Systems", "I/O Management"): [
        (WIKI, "https://en.wikipedia.org/wiki/Input/output"),
        (TUTORIALS, "https://www.tutorialspoint.com/operating_system/os_io_software.htm"),
    ],
    ("Operating Systems", "Security"): [
        (WIKI, "https://en.wikipedia.org/wiki/Operating-system-level_virtualization"),
        (WIKI, "https://en.wikipedia.org/wiki/Security-Enhanced_Linux"),
    ],
    # ---------------- Computer Networks ----------------
    ("Computer Networks", "OSI"): [
        (WIKI, "https://en.wikipedia.org/wiki/OSI_model"),
        (GEEKS, "https://www.geeksforgeeks.org/open-systems-interconnection-model-osi/"),
    ],
    ("Computer Networks", "TCP/IP"): [
        (WIKI, "https://en.wikipedia.org/wiki/Internet_protocol_suite"),
        (GEEKS, "https://www.geeksforgeeks.org/tcp-ip-model/"),
    ],
    ("Computer Networks", "IP Addressing"): [
        (WIKI, "https://en.wikipedia.org/wiki/IP_address"),
        (WIKI, "https://en.wikipedia.org/wiki/Subnetwork"),
    ],
    ("Computer Networks", "Routing"): [
        (WIKI, "https://en.wikipedia.org/wiki/Routing"),
    ],
    ("Computer Networks", "TCP"): [
        (WIKI, "https://en.wikipedia.org/wiki/Transmission_Control_Protocol"),
    ],
    ("Computer Networks", "UDP"): [
        (WIKI, "https://en.wikipedia.org/wiki/User_Datagram_Protocol"),
        (GEEKS, "https://www.geeksforgeeks.org/user-datagram-protocol-udp/"),
    ],
    ("Computer Networks", "HTTP"): [
        (WIKI, "https://en.wikipedia.org/wiki/HTTP"),
        (MDN, "https://developer.mozilla.org/en-US/docs/Web/HTTP"),
    ],
    ("Computer Networks", "DNS"): [
        (WIKI, "https://en.wikipedia.org/wiki/Domain_Name_System"),
        (GEEKS, "https://www.geeksforgeeks.org/domain-name-system-dns-in-application-layer/"),
    ],
    ("Computer Networks", "DHCP"): [
        (WIKI, "https://en.wikipedia.org/wiki/Dynamic_Host_Configuration_Protocol"),
        (GEEKS, "https://www.geeksforgeeks.org/dynamic-host-configuration-protocol-dhcp/"),
    ],
    ("Computer Networks", "Network Security"): [
        (WIKI, "https://en.wikipedia.org/wiki/Network_security"),
        (WIKI, "https://en.wikipedia.org/wiki/Firewall_(computing)"),
    ],
    # ---------------- Software Engineering ----------------
    ("Software Engineering", "SDLC"): [
        (WIKI, "https://en.wikipedia.org/wiki/Systems_development_life_cycle"),
    ],
    ("Software Engineering", "Waterfall"): [
        (WIKI, "https://en.wikipedia.org/wiki/Waterfall_model"),
        (TUTORIALS, "https://www.tutorialspoint.com/sdlc/sdlc_waterfall_model.htm"),
    ],
    ("Software Engineering", "Agile"): [
        (WIKI, "https://en.wikipedia.org/wiki/Agile_software_development"),
    ],
    ("Software Engineering", "SRS"): [
        (WIKI, "https://en.wikipedia.org/wiki/Software_requirements_specification"),
    ],
    ("Software Engineering", "Design Patterns"): [
        (WIKI, "https://en.wikipedia.org/wiki/Software_design_pattern"),
        (GEEKS, "https://www.geeksforgeeks.org/design-patterns-in-java/"),
    ],
    ("Software Engineering", "UML"): [
        (WIKI, "https://en.wikipedia.org/wiki/Unified_Modeling_Language"),
        (GEEKS, "https://www.geeksforgeeks.org/unified-modeling-language-uml-introduction/"),
    ],
    ("Software Engineering", "Testing"): [
        (WIKI, "https://en.wikipedia.org/wiki/Software_testing"),
        (GEEKS, "https://www.geeksforgeeks.org/software-testing-basics/"),
    ],
    ("Software Engineering", "Software Quality"): [
        (WIKI, "https://en.wikipedia.org/wiki/Software_quality"),
        (WIKI, "https://en.wikipedia.org/wiki/ISO/IEC_9126"),
    ],
    ("Software Engineering", "Project Management"): [
        (WIKI, "https://en.wikipedia.org/wiki/Software_project_management"),
        (WIKI, "https://en.wikipedia.org/wiki/Project_management"),
    ],
    ("Software Engineering", "Git/Version Control"): [
        (WIKI, "https://en.wikipedia.org/wiki/Git"),
        (WIKI, "https://en.wikipedia.org/wiki/Version_control"),
    ],
}


def get_links(subject: str, topic: str) -> List[Link]:
    return REFERENCES.get((subject, topic), [])
