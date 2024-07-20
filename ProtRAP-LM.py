from models import *

def run_ProtRAP_LM(fasta_file, result_dir, device):

    seq = fasta_load(fasta_file)
    result=ProtRAP_LM(device).predict(seq)
    np.savetxt(result_dir + 'results.txt', np.column_stack((result[:,0:1], result[:,1:2], result[:,0:1]*result[:,1:2], (1-result[:,0:1])*result[:,1:2])), fmt='%.4f %.4f %.4f %.4f', header='MCP RASA RLA RSA')

if __name__=='__main__':
    argparser=argparse.ArgumentParser()
    argparser.add_argument('--input_file', help='input file')
    argparser.add_argument('--output_dir', default='./', help='output directory')
    argparser.add_argument('--device', default='cpu', help='which device')
    args=argparser.parse_args()
    
    run_ProtRAP_LM(str(args.input_file), str(args.output_path), str(args.device) )
