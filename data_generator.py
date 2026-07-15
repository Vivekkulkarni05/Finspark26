import pandas as pd
import numpy as np
import datetime
import random
import uuid

def generate_data(num_records=2000):
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # 1. Base entities
    num_users = int(num_records / 10)
    users = [f"U{str(i).zfill(5)}" for i in range(num_users)]
    devices = [f"DEV-{uuid.uuid4().hex[:8]}" for _ in range(num_users * 2)]
    ips = [f"{random.randint(10, 192)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}" for _ in range(num_users * 3)]
    accounts = [f"ACC-{str(i).zfill(6)}" for i in range(num_users * 2)]

    # Time configuration
    start_time = datetime.datetime(2026, 1, 1, 12, 0, 0)
    
    # --- Generate Transactions (PaySim-like) ---
    tx_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    tx_data = []
    
    for i in range(num_records):
        timestamp = start_time + datetime.timedelta(minutes=random.randint(0, 10000))
        tx_type = random.choice(tx_types)
        amount = round(random.uniform(10, 1500), 2)
        user_id = random.choice(users)
        account_id = random.choice(accounts)
        device_id = random.choice(devices)
        ip = random.choice(ips)
        oldbalanceOrg = round(random.uniform(100, 50000), 2)
        newbalanceOrig = oldbalanceOrg - amount if tx_type in ['PAYMENT', 'TRANSFER', 'CASH_OUT'] else oldbalanceOrg + amount
        
        tx_data.append({
            'event_id': f"TX-{uuid.uuid4().hex[:8]}",
            'timestamp': timestamp,
            'type': tx_type,
            'amount': amount,
            'nameOrig': user_id,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'nameDest': f"U{random.randint(10000, 99999)}",
            'isFraud': 0,
            'user_id': user_id,
            'device_id': device_id,
            'ip': ip,
            'account_id': account_id
        })
    
    df_tx = pd.DataFrame(tx_data)

    # --- Generate Cyber Events (CICIDS2017-like) ---
    protocols = ['TCP', 'UDP', 'HTTP']
    tls_versions = ['TLSv1.2', 'TLSv1.3', 'None', 'TLSv1.1']
    cipher_suites = ['TLS_AES_128_GCM_SHA256', 'TLS_AES_256_GCM_SHA384', 'TLS_RSA_WITH_AES_128_CBC_SHA', 'None']
    labels = ['BENIGN', 'DDoS', 'BruteForce', 'PortScan']
    
    net_data = []
    
    for i in range(num_records):
        # We link ~20% of network events to existing users/devices/ips within a close time window of a transaction
        if random.random() < 0.2:
            base_tx = random.choice(tx_data)
            timestamp = base_tx['timestamp'] - datetime.timedelta(minutes=random.randint(1, 10))
            user_id = base_tx['user_id']
            device_id = base_tx['device_id']
            ip = base_tx['ip']
        else:
            timestamp = start_time + datetime.timedelta(minutes=random.randint(0, 10000))
            user_id = random.choice(users)
            device_id = random.choice(devices)
            ip = random.choice(ips)
            
        protocol = random.choice(protocols)
        tls_ver = random.choice(tls_versions) if protocol in ['TCP', 'HTTP'] else 'None'
        cipher = random.choice(cipher_suites) if tls_ver != 'None' else 'None'
        flow_duration = random.randint(10, 15000) # milliseconds
        
        net_data.append({
            'event_id': f"NET-{uuid.uuid4().hex[:8]}",
            'timestamp': timestamp,
            'src_ip': ip,
            'dst_ip': '10.0.0.1', # Server IP
            'protocol': protocol,
            'flow_duration': flow_duration,
            'tls_version': tls_ver,
            'cipher_suite': cipher,
            'bytes_sent': random.randint(100, 5000),
            'bytes_received': random.randint(100, 5000),
            'label': 'BENIGN',
            'user_id': user_id,
            'device_id': device_id,
            'ip': ip,
            'isFraud': 0, # To unify later
            'event_type': 'network_log'
        })
        
    df_net = pd.DataFrame(net_data)

    # --- Scenario 1: Account Takeover ---
    # Failed login, then successful login (network events), followed by a high-value transaction on a new device/IP.
    ato_user = users[0]
    ato_acc = accounts[0]
    ato_time = start_time + datetime.timedelta(days=1)
    ato_new_device = "DEV-HACKER01"
    ato_new_ip = "198.51.100.22"
    
    df_net = pd.concat([df_net, pd.DataFrame([{
        'event_id': f"NET-ATO-FAIL", 'timestamp': ato_time, 'src_ip': ato_new_ip, 'dst_ip': '10.0.0.1',
        'protocol': 'TCP', 'flow_duration': 120, 'tls_version': 'TLSv1.3', 'cipher_suite': 'TLS_AES_128_GCM_SHA256',
        'bytes_sent': 500, 'bytes_received': 200, 'label': 'BruteForce_Failed',
        'user_id': ato_user, 'device_id': ato_new_device, 'ip': ato_new_ip, 'isFraud': 1, 'event_type': 'login_failed'
    }])], ignore_index=True)
    
    df_net = pd.concat([df_net, pd.DataFrame([{
        'event_id': f"NET-ATO-SUCC", 'timestamp': ato_time + datetime.timedelta(minutes=2), 'src_ip': ato_new_ip, 'dst_ip': '10.0.0.1',
        'protocol': 'TCP', 'flow_duration': 150, 'tls_version': 'TLSv1.3', 'cipher_suite': 'TLS_AES_128_GCM_SHA256',
        'bytes_sent': 600, 'bytes_received': 250, 'label': 'BruteForce_Success',
        'user_id': ato_user, 'device_id': ato_new_device, 'ip': ato_new_ip, 'isFraud': 1, 'event_type': 'login_success'
    }])], ignore_index=True)
    
    df_tx = pd.concat([df_tx, pd.DataFrame([{
        'event_id': f"TX-ATO-HIGH", 'timestamp': ato_time + datetime.timedelta(minutes=5), 'type': 'TRANSFER', 'amount': 45000.0,
        'nameOrig': ato_user, 'oldbalanceOrg': 50000.0, 'newbalanceOrig': 5000.0, 'nameDest': 'U-HACKER',
        'isFraud': 1, 'user_id': ato_user, 'device_id': ato_new_device, 'ip': ato_new_ip, 'account_id': ato_acc
    }])], ignore_index=True)

    # --- Scenario 2: Fraud Ring ---
    # 1 device_id linked to 5+ distinct account_ids transacting in a burst.
    ring_device = "DEV-RINGMASTER"
    ring_ip = "203.0.113.55"
    ring_time = start_time + datetime.timedelta(days=2)
    ring_users = users[10:15]
    ring_accs = accounts[10:15]
    
    for i in range(5):
        df_tx = pd.concat([df_tx, pd.DataFrame([{
            'event_id': f"TX-RING-{i}", 'timestamp': ring_time + datetime.timedelta(minutes=i), 'type': 'CASH_OUT', 'amount': 9500.0,
            'nameOrig': ring_users[i], 'oldbalanceOrg': 10000.0, 'newbalanceOrig': 500.0, 'nameDest': 'U-RINGDROP',
            'isFraud': 1, 'user_id': ring_users[i], 'device_id': ring_device, 'ip': ring_ip, 'account_id': ring_accs[i]
        }])], ignore_index=True)
        # Network event for each
        df_net = pd.concat([df_net, pd.DataFrame([{
            'event_id': f"NET-RING-{i}", 'timestamp': ring_time + datetime.timedelta(minutes=i) - datetime.timedelta(seconds=30), 
            'src_ip': ring_ip, 'dst_ip': '10.0.0.1', 'protocol': 'HTTP', 'flow_duration': 800, 'tls_version': 'TLSv1.2', 
            'cipher_suite': 'TLS_AES_128_GCM_SHA256', 'bytes_sent': 1500, 'bytes_received': 3000, 'label': 'BENIGN',
            'user_id': ring_users[i], 'device_id': ring_device, 'ip': ring_ip, 'isFraud': 1, 'event_type': 'session'
        }])], ignore_index=True)

    # --- Scenario 3: Quantum-exposure case ---
    # Large one-way encrypted data transfer using TLS <1.2 or RSA/ECC without PFS, 
    # no corresponding user activity afterward, paired with an unrelated small "cover" transaction.
    q_user = users[20]
    q_acc = accounts[20]
    q_device = devices[20]
    q_ip = ips[20]
    q_time = start_time + datetime.timedelta(days=3)
    
    df_net = pd.concat([df_net, pd.DataFrame([{
        'event_id': f"NET-QEXFIL", 'timestamp': q_time, 'src_ip': q_ip, 'dst_ip': '10.0.0.1',
        'protocol': 'TCP', 'flow_duration': 300000, 'tls_version': 'TLSv1.1', 'cipher_suite': 'TLS_RSA_WITH_AES_128_CBC_SHA', # No PFS, TLS < 1.2
        'bytes_sent': 50000000, 'bytes_received': 500, 'label': 'BENIGN', # archival exfil pattern
        'user_id': q_user, 'device_id': q_device, 'ip': q_ip, 'isFraud': 1, 'event_type': 'data_transfer'
    }])], ignore_index=True)
    
    df_tx = pd.concat([df_tx, pd.DataFrame([{
        'event_id': f"TX-QCOVER", 'timestamp': q_time + datetime.timedelta(minutes=5), 'type': 'PAYMENT', 'amount': 15.0,
        'nameOrig': q_user, 'oldbalanceOrg': 1200.0, 'newbalanceOrig': 1185.0, 'nameDest': 'U-STORE',
        'isFraud': 1, 'user_id': q_user, 'device_id': q_device, 'ip': q_ip, 'account_id': q_acc
    }])], ignore_index=True)

    return df_tx, df_net

def join_datasets(df_tx, df_net, window_minutes=15):
    """
    Join transaction and network events based on shared IDs (user, device, IP) within a time window.
    """
    # Sort by timestamp
    df_tx = df_tx.sort_values('timestamp')
    df_net = df_net.sort_values('timestamp')
    
    # We want a unified event table. One approach:
    # Merge on (user_id, device_id, ip).
    # pandas merge_asof is perfect for joining nearest timestamp.
    
    merged = pd.merge_asof(
        df_tx, df_net,
        on='timestamp',
        by=['user_id', 'device_id', 'ip'],
        direction='backward',
        tolerance=pd.Timedelta(minutes=window_minutes)
    )
    
    # Fill NAs
    merged['tls_version'] = merged['tls_version'].fillna('Unknown')
    merged['cipher_suite'] = merged['cipher_suite'].fillna('Unknown')
    merged['bytes_sent'] = merged['bytes_sent'].fillna(0)
    merged['bytes_received'] = merged['bytes_received'].fillna(0)
    merged['label'] = merged['label'].fillna('Unknown')
    
    # Determine overall fraud (if either was labeled fraud)
    merged['isFraud'] = merged['isFraud_x'] | merged['isFraud_y'].fillna(0).astype(int)
    merged.drop(['isFraud_x', 'isFraud_y'], axis=1, inplace=True)
    
    # Calculate User Transaction/Activity Velocity (Events in the last 1 hour)
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    
    velocity = []
    for i in range(len(merged)):
        row = merged.iloc[i]
        user = row['user_id']
        ts = row['timestamp']
        # Count events for this user in the last 1 hour up to the current timestamp
        count = ((merged['user_id'] == user) & (merged['timestamp'] <= ts) & (merged['timestamp'] >= ts - pd.Timedelta(hours=1))).sum()
        velocity.append(count)
    merged['user_tx_velocity'] = velocity
    
    return merged

if __name__ == "__main__":
    print("Generating synthetic datasets...")
    df_tx, df_net = generate_data(num_records=2000)
    print(f"Generated {len(df_tx)} transactions and {len(df_net)} network events.")
    
    print("Joining datasets...")
    joined_df = join_datasets(df_tx, df_net, window_minutes=15)
    print(f"Joined table shape: {joined_df.shape}")
    
    # Check scenarios
    print("\nScenario Verification:")
    print("ATO Transactions:", joined_df[joined_df['event_id_x'] == 'TX-ATO-HIGH'][['event_id_x', 'event_id_y', 'isFraud']])
    print("Fraud Ring Transactions:", joined_df[joined_df['event_id_x'].str.startswith('TX-RING')][['event_id_x', 'event_id_y', 'isFraud']])
    print("Quantum Exfil Transactions:", joined_df[joined_df['event_id_x'] == 'TX-QCOVER'][['event_id_x', 'event_id_y', 'isFraud']])
    
    joined_df.to_csv("joined_data.csv", index=False)
    print("\nSaved to joined_data.csv")
