"""ICT vocabulary enhancement.

1. Recognition hints: a compact initial prompt that biases Whisper toward
   correct spellings of technical terms.
2. Post-recognition normalisation: deliberately conservative. A rewrite is
   applied only when the recognised letters are *identical* to a known term
   (differences in case, spacing, dots or hyphens only: "v lan" -> "VLAN",
   "d.h.c.p" -> "DHCP", "hyper v" -> "Hyper-V"). It never substitutes a
   different word, so it cannot change the user's meaning. Every change is
   returned so the UI can show what was heard versus what was normalised.
3. Pronunciation lexicon so TTS reads abbreviations properly.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# Canonical spellings. Acronyms/proper nouns are normalised; common words are hints only.
ACRONYMS = [
    "VLAN", "DNS", "DHCP", "VPN", "SIEM", "EDR", "XDR", "MFA", "RBAC", "ITIL", "COBIT", "LAN", "WAN", "SD-WAN",
    "NAT", "ACL", "BGP", "OSPF", "TCP", "UDP", "IP", "IPv4", "IPv6", "HTTP", "HTTPS", "TLS", "SSL", "SSH", "RDP",
    "SMB", "LDAP", "SAML", "OAuth", "SSO", "IAM", "PAM", "SOC", "NOC", "CVE", "IDS", "IPS", "WAF", "DLP", "UTM",
    "NGFW", "SASE", "ZTNA", "API", "CPU", "GPU", "RAM", "SSD", "NAS", "SAN", "RAID", "VM", "AWS", "GCP", "M365",
    "GPO", "OU", "PKI", "SNMP", "NTP", "SMTP", "IMAP", "ISP", "QoS", "PoE", "SFP", "MAC", "ARP", "ICMP", "SIP",
    "VoIP", "CRM", "ERP", "SLA", "KPI", "CMDB", "ITSM", "NIST", "GDPR", "PCI DSS", "HIPAA", "SOAR", "MDM", "EMM",
    "UEM", "BYOD", "RPO", "RTO", "DR", "BCP", "AI", "ML", "LLM", "RAG", "NVE",
]
PROPER = [
    "Active Directory", "Microsoft Entra ID", "Entra ID", "Azure AD", "Intune", "Windows Server", "Linux", "Ubuntu",
    "VMware", "vSphere", "ESXi", "Hyper-V", "Microsoft 365", "Exchange Online", "SharePoint", "Teams", "Azure",
    "Endpoint Detection and Response", "ISO 27001", "ISO 9001", "Fortinet", "FortiGate", "Cisco", "Palo Alto",
    "Sophos", "Kaspersky", "CrowdStrike", "Defender for Endpoint", "PowerShell", "Docker", "Kubernetes", "Ollama",
    "Wireshark", "Nmap", "pfSense", "MikroTik", "Proxmox", "Veeam", "Splunk", "Wazuh", "N.I.M.R.A.S.",
]
COMMON_HINTS = ["firewall", "router", "switch", "server", "endpoint", "backup", "patching", "subnet", "gateway",
                "domain controller", "group policy", "load balancer", "access point", "encryption", "malware",
                "phishing", "ransomware", "vulnerability", "penetration testing", "incident response"]

# How TTS should say things (English voice). Keys are matched case-sensitively as whole words.
PRONUNCIATION = {
    "VLAN": "V-LAN", "SIEM": "seem", "ITIL": "eye-till", "COBIT": "co-bit", "RBAC": "R-back", "MFA": "M F A",
    "DHCP": "D H C P", "DNS": "D N S", "VPN": "V P N", "EDR": "E D R", "XDR": "X D R", "ISO": "eye-so",
    "NAS": "nass", "SAN": "san", "RAID": "raid", "SASE": "sassy", "ZTNA": "Z T N A", "SOAR": "soar",
    "VMware": "V M ware", "ESXi": "E S X i", "Hyper-V": "Hyper V", "Entra": "Entra", "Intune": "In-tune",
    "SQL": "sequel", "GUI": "gooey", "SCADA": "skada", "N.I.M.R.A.S.": "Nimras", "NIMRAS": "Nimras",
    "NGFW": "next generation firewall", "OAuth": "oh-auth", "IPv4": "I P v 4", "IPv6": "I P v 6",
    "PoE": "P O E", "QoS": "Q O S", "VoIP": "voyp", "LLM": "L L M", "RAG": "rag", "NVE": "N V E",
    "SD-WAN": "S D WAN", "Wi-Fi": "why-fye", "SharePoint": "Share Point", "PowerShell": "Power Shell",
}


# How acronyms are commonly split when spoken (letters identical, so still safe).
SPOKEN_SPLITS = {
    "VLAN": [("v", "lan")], "SD-WAN": [("sd", "wan")], "IPv4": [("ip", "v4")], "IPv6": [("ip", "v6")],
    "VoIP": [("vo", "ip")], "NGFW": [("ng", "fw")], "M365": [("m", "365")], "OAuth": [("o", "auth")],
}


def _letters(term: str) -> str:
    return re.sub(r"[^0-9a-z]", "", term.lower())


class Vocabulary:
    def __init__(self, extra_file: Path | None = None):
        self.acronyms = list(ACRONYMS)
        self.proper = list(PROPER)
        self.hints = list(COMMON_HINTS)
        self.pronunciation = dict(PRONUNCIATION)
        if extra_file and extra_file.exists():
            try:
                extra = json.loads(extra_file.read_text(encoding="utf-8"))
                self.acronyms += extra.get("acronyms", [])
                self.proper += extra.get("proper", [])
                self.hints += extra.get("hints", [])
                self.pronunciation.update(extra.get("pronunciation", {}))
            except (ValueError, OSError):
                pass
        self._rules = self._build_rules()

    # --------------------------------------------------------- recognition hints
    def initial_prompt(self, recent_text: str = "", max_chars: int = 260) -> str:
        """Whisper's prompt is limited (~224 tokens); keep it short and relevant.
        Terms that appeared in the recent conversation are listed first."""
        recent = recent_text.lower()
        terms = self.proper + self.acronyms
        terms.sort(key=lambda t: (t.lower() not in recent, len(t)))
        head = "IT support. Terms: "
        out = head
        for t in terms + self.hints:
            if len(out) + len(t) + 2 > max_chars:
                break
            out += t + ", "
        return out.rstrip(", ") + "."

    # --------------------------------------------------------- normalisation
    def _build_rules(self):
        rules = []
        for term in sorted(set(self.acronyms + self.proper), key=len, reverse=True):
            letters = _letters(term)
            if len(letters) < 2:
                continue
            if term in self.acronyms and " " not in term:
                # contiguous ("dhcp") or spelled out letter by letter ("d h c p", "d.h.c.p.")
                # Either the whole word, or EVERY letter separated ("d h c p", "d.h.c.p.") -
                # never a partial mix, so "I am" can't become "IAM".
                spelled = r"[\s.\-]".join(re.escape(c) for c in letters)
                alts = re.escape(letters) + (f"|{spelled}" if len(letters) <= 6 else "")
                for split in SPOKEN_SPLITS.get(term, []):          # e.g. "v lan" for VLAN
                    alts += "|" + r"[\s.\-]?".join(re.escape(p) for p in split)
                pat = rf"(?<![\w.])(?:{alts})\.?(?![\w])"
            else:
                words = re.findall(r"[0-9A-Za-z]+", term)
                pat = r"\b" + r"[\s\-.]*".join(re.escape(w) for w in words) + r"\b"
            rules.append((re.compile(pat, re.I), term))
        return rules

    def normalize(self, text: str) -> tuple[str, list[dict]]:
        """Returns (normalised_text, changes). Only case/spacing/punctuation of
        known terms is altered; letters are never changed."""
        changes: list[dict] = []
        taken: list[tuple[int, int]] = []
        repl: list[tuple[int, int, str]] = []
        for rx, term in self._rules:
            for m in rx.finditer(text):
                a, b = m.span()
                if any(a < y and x < b for x, y in taken):
                    continue
                heard = m.group(0)
                if _letters(heard) != _letters(term):
                    continue
                # Ambiguity guard: lowercase, unspaced short acronyms that are
                # also English words (e.g. "it", "am", "soar") are left alone.
                if heard.islower() and heard.isalpha() and len(heard) <= 4 and heard in _ENGLISH_WORDS:
                    continue
                taken.append((a, b))
                if heard != term:
                    repl.append((a, b, term))
                    changes.append({"heard": heard, "normalized": term})
        for a, b, term in sorted(repl, reverse=True):
            text = text[:a] + term + text[b:]
        return text, changes

    # --------------------------------------------------------- TTS
    def for_speech(self, text: str) -> str:
        def sub(m):
            return self.pronunciation.get(m.group(0), m.group(0))
        keys = sorted(self.pronunciation, key=len, reverse=True)
        rx = re.compile(r"(?<![\w-])(" + "|".join(re.escape(k) for k in keys) + r")(?![\w-])")
        text = rx.sub(sub, text)
        # Markdown and symbols TTS would read literally.
        text = re.sub(r"```.*?```", " (code omitted) ", text, flags=re.S)
        text = re.sub(r"[*_`#>|]+", " ", text)
        text = re.sub(r"https?://\S+", " a link ", text)
        text = text.replace("&", " and ").replace("/", " slash ") if re.search(r"\w/\w", text) else text.replace("&", " and ")
        return re.sub(r"\s{2,}", " ", text).strip()


_ENGLISH_WORDS = {"it", "am", "ai", "ip", "ml", "dr", "soar", "san", "mac", "sip", "ram", "pam", "nat", "rag",
                  "ou", "api", "ops", "soc", "ids", "ips", "dlp", "acl", "sso", "isp", "bcp", "rto", "rpo",
                  "vm", "lan", "wan", "noc", "emm", "sla", "kpi", "erp", "crm", "teams", "linux", "azure",
                  "docker", "ollama", "cisco", "sophos", "splunk", "wazuh"}
