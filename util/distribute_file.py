import os
import glob
import paramiko

class PythonFileDistributor:
    def __init__(self, local_directory, remote_directory, nodes):
        self.local_directory = local_directory
        self.remote_directory = remote_directory
        self.nodes = nodes

    def get_python_files(self):
        return glob.glob(os.path.join(self.local_directory, '*.py'))

    def connect_to_node(self, node_info):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if 'private_key' in node_info:
            key = paramiko.RSAKey.from_private_key_file(node_info['private_key'])
            ssh.connect(node_info['hostname'], username=node_info['username'], pkey=key)
        else:
            ssh.connect(node_info['hostname'], username=node_info['username'], password=node_info['password'])
        return ssh

    def transfer_file(self, ssh, localfile, remotefile):
        sftp = ssh.open_sftp()
        sftp.put(localfile, remotefile)
        sftp.close()

    def distribute_files(self):
        python_files = self.get_python_files()
        for node in self.nodes:
            try:
                print(f"Connecting to {node['hostname']}")
                ssh = self.connect_to_node(node)

                for pyfile in python_files:
                    remote_path = os.path.join(self.remote_directory, os.path.basename(pyfile))
                    print(f"Transferring {pyfile} to {node['hostname']}:{remote_path}")
                    self.transfer_file(ssh, pyfile, remote_path)

                ssh.close()
                print(f"Finished transferring files to {node['hostname']}")
            except Exception as e:
                print(f"Failed to transfer files to {node['hostname']}: {e}")


if __name__ == '__main__':
    nodes = [
        {'hostname': '', 'username': '', 'password': ''},
        {'hostname': '', 'username': '', 'password': ''},
        {'hostname': '', 'username': '', 'password': ''},
        {'hostname': '', 'username': '', 'password': ''}
    ]
    distributor = PythonFileDistributor(
        local_directory='',
        remote_directory='',
        nodes=nodes
    )
    distributor.distribute_files()