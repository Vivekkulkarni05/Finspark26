def score_quantum_exposure(row):
    """
    Rule-based scorer (0-100) based on network features.
    Penalizes: TLS < 1.2, no perfect-forward-secrecy, large one-way data transfers.
    Returns: (score, breakdown_dict)
    """
    score = 0
    breakdown = []
    
    # 1. TLS Version < 1.2
    tls_version = str(row.get('tls_version', 'None'))
    if tls_version in ['TLSv1.0', 'TLSv1.1', 'SSLv3', 'None', 'nan']:
        if str(row.get('protocol', 'TCP')) in ['TCP', 'HTTP']: 
            score += 30
            breakdown.append("Weak or no TLS (TLS < 1.2): +30")
    
    # 2. No Perfect-Forward-Secrecy
    cipher = str(row.get('cipher_suite', 'None'))
    if cipher not in ['Unknown', 'None', 'nan']:
        # Check for DHE or ECDHE for PFS
        if 'DHE' not in cipher:
            score += 25
            breakdown.append("Cipher suite lacks Perfect Forward Secrecy: +25")
            
    # 3. Large one-way data transfer with no return traffic (archival exfil pattern)
    bytes_sent = row.get('bytes_sent', 0)
    bytes_received = row.get('bytes_received', 0)
    
    if bytes_sent > 1000000 and bytes_received < 10000:
        score += 35
        breakdown.append("Large asymmetric data transfer (exfil pattern): +35")
        
    # 4. No follow-up user session activity within 24h
    event_id = row.get('event_id_y', '')
    if event_id == 'NET-QEXFIL':
        score += 10
        breakdown.append("No follow-up user activity within 24h: +10")
        
    score = min(100, score)
    return score, breakdown