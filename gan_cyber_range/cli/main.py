"""
Main CLI entry point for GAN-Cyber-Range-v2.

Provides comprehensive command-line interface for all cyber range operations
including training, deployment, and scenario execution.
"""

import click
import asyncio
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any

from ..core.cyber_range import CyberRange
from ..core.network_sim import NetworkTopology
from ..core.attack_gan import AttackGAN
from ..generators.malware_gan import MalwareGAN
from ..generators.network_gan import NetworkAttackGAN
from ..generators.web_attack_gan import WebAttackGAN
from ..utils.logging_config import setup_logging
from ..utils.validation import validate_config

logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-file', type=click.Path(), help='Log file path')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool, log_file: Optional[str]):
    """
    GAN-Cyber-Range-v2: Advanced Cybersecurity Training Platform
    
    A second-generation adversarial cyber range that combines GAN-based attack generation 
    with LLM-driven red team curricula for comprehensive cybersecurity training and research.
    """
    # Ensure context exists
    ctx.ensure_object(dict)
    
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=log_level, log_file=log_file)
    
    # Load configuration
    if config:
        with open(config, 'r') as f:
            ctx.obj['config'] = json.load(f)
    else:
        ctx.obj['config'] = {}
        
    logger.info("GAN-Cyber-Range-v2 CLI initialized")


@cli.command()
@click.option('--topology', '-t', type=click.Choice(['small', 'medium', 'large', 'enterprise']), 
              default='medium', help='Network topology size')
@click.option('--hosts', type=int, help='Number of hosts to create')
@click.option('--subnets', type=str, help='Comma-separated list of subnet names')
@click.option('--output', '-o', type=click.Path(), help='Output topology file')
@click.pass_context
def create_topology(ctx, topology: str, hosts: Optional[int], subnets: Optional[str], output: Optional[str]):
    """Create network topology for cyber range."""
    
    try:
        logger.info(f"Creating {topology} network topology")
        
        # Configure topology parameters
        if topology == 'small':
            hosts_per_subnet = {'dmz': 2, 'internal': 5, 'management': 2}
            subnet_list = ['dmz', 'internal', 'management']
        elif topology == 'medium':
            hosts_per_subnet = {'dmz': 5, 'internal': 20, 'management': 5}
            subnet_list = ['dmz', 'internal', 'management', 'development']
        elif topology == 'large':
            hosts_per_subnet = {'dmz': 10, 'internal': 50, 'management': 8, 'development': 15}
            subnet_list = ['dmz', 'internal', 'management', 'development', 'guest']
        else:  # enterprise
            hosts_per_subnet = {'dmz': 15, 'internal': 100, 'management': 12, 'development': 25, 'guest': 10}
            subnet_list = ['dmz', 'internal', 'management', 'development', 'guest', 'iot']
            
        # Override with custom parameters
        if subnets:
            subnet_list = [s.strip() for s in subnets.split(',')]
            
        if hosts:
            # Distribute hosts evenly across subnets
            hosts_per_subnet = {subnet: hosts // len(subnet_list) for subnet in subnet_list}
            
        # Generate topology
        net_topology = NetworkTopology.generate(
            template=topology,
            subnets=subnet_list,
            hosts_per_subnet=hosts_per_subnet,
            services=['web', 'database', 'email', 'file_share', 'ssh', 'rdp'],
            vulnerabilities='realistic'
        )
        
        click.echo(f"✅ Created topology with {net_topology.total_hosts} hosts across {len(net_topology.subnets)} subnets")
        
        # Display topology summary
        for subnet in net_topology.subnets:
            host_count = len(net_topology.get_hosts_by_subnet(subnet.name))
            click.echo(f"  📋 {subnet.name}: {host_count} hosts ({subnet.cidr})")
            
        # Save topology if output specified
        if output:
            net_topology.save_to_file(output)
            click.echo(f"💾 Topology saved to {output}")
            
    except Exception as e:
        logger.error(f"Failed to create topology: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--topology-file', '-t', type=click.Path(exists=True), 
              help='Topology JSON file')
@click.option('--name', '-n', type=str, required=True,
              help='Cyber range name')
@click.option('--resource-limits', type=str, 
              help='Resource limits JSON (e.g., \'{"cpu_cores": 8, "memory_gb": 16}\')')
@click.option('--monitoring/--no-monitoring', default=True,
              help='Enable monitoring')
@click.option('--isolation', type=click.Choice(['container', 'vm', 'strict']),
              default='container', help='Isolation level')
@click.pass_context
def deploy(ctx, topology_file: Optional[str], name: str, resource_limits: Optional[str], 
           monitoring: bool, isolation: str):
    """Deploy cyber range infrastructure."""
    
    try:
        logger.info(f"Deploying cyber range: {name}")
        
        # Load or create topology
        if topology_file:
            topology = NetworkTopology.load_from_file(topology_file)
            click.echo(f"📁 Loaded topology from {topology_file}")
        else:
            # Create default medium topology
            topology = NetworkTopology.generate(
                template='medium',
                subnets=['dmz', 'internal', 'management'],
                hosts_per_subnet={'dmz': 5, 'internal': 20, 'management': 3}
            )
            click.echo("🏗️  Created default medium topology")
            
        # Parse resource limits
        limits = {}
        if resource_limits:
            limits = json.loads(resource_limits)
            
        # Create cyber range
        cyber_range = CyberRange(
            topology=topology,
            hypervisor='docker',
            container_runtime='docker',
            network_emulation='bridge'
        )
        
        # Deploy infrastructure
        click.echo("🚀 Deploying cyber range infrastructure...")
        
        with click.progressbar(length=100, label='Deploying') as bar:
            # Simulate deployment progress
            range_id = cyber_range.deploy(
                resource_limits=limits,
                isolation_level=isolation,
                monitoring=monitoring
            )
            bar.update(50)
            
            # Start the range
            cyber_range.start()
            bar.update(100)
            
        click.echo(f"✅ Cyber range '{name}' deployed successfully!")
        click.echo(f"🆔 Range ID: {range_id}")
        click.echo(f"🌐 Dashboard: {cyber_range.dashboard_url}")
        click.echo(f"🔐 VPN Config: {cyber_range.vpn_config_path}")
        
        # Save range info for later use
        range_info = {
            'name': name,
            'range_id': range_id,
            'dashboard_url': cyber_range.dashboard_url,
            'vpn_config_path': cyber_range.vpn_config_path,
            'deployment_time': cyber_range.start_time.isoformat() if cyber_range.start_time else None
        }
        
        info_file = Path(f"{name}_range_info.json")
        with open(info_file, 'w') as f:
            json.dump(range_info, f, indent=2)
            
        click.echo(f"📋 Range info saved to {info_file}")
        
    except Exception as e:
        logger.error(f"Failed to deploy cyber range: {e}")
        click.echo(f"❌ Deployment failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--gan-type', '-g', type=click.Choice(['attack', 'malware', 'network', 'web']),
              required=True, help='Type of GAN to train')
@click.option('--data-path', '-d', type=click.Path(exists=True), required=True,
              help='Training data directory')
@click.option('--epochs', '-e', type=int, default=1000,
              help='Number of training epochs')
@click.option('--batch-size', '-b', type=int, default=64,
              help='Training batch size')
@click.option('--output-dir', '-o', type=click.Path(), default='./models',
              help='Output directory for trained models')
@click.option('--gpu/--no-gpu', default=True,
              help='Use GPU acceleration if available')
@click.pass_context
def train(ctx, gan_type: str, data_path: str, epochs: int, batch_size: int, 
          output_dir: str, gpu: bool):
    """Train GAN models for attack generation."""
    
    try:
        logger.info(f"Training {gan_type} GAN with {epochs} epochs")
        
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        device = "cuda" if gpu else "cpu"
        
        # Initialize appropriate GAN
        if gan_type == 'attack':
            gan = AttackGAN(device=device)
            click.echo("🧠 Initializing Attack GAN")
        elif gan_type == 'malware':
            gan = MalwareGAN(device=device)
            click.echo("🦠 Initializing Malware GAN")
        elif gan_type == 'network':
            gan = NetworkAttackGAN(device=device)
            click.echo("🌐 Initializing Network Attack GAN")
        elif gan_type == 'web':
            gan = WebAttackGAN(device=device)
            click.echo("🕸️  Initializing Web Attack GAN")
        else:
            raise ValueError(f"Unknown GAN type: {gan_type}")
            
        # Load training data
        click.echo(f"📁 Loading training data from {data_path}")
        training_data = _load_training_data(data_path, gan_type)
        
        click.echo(f"📊 Loaded {len(training_data)} training samples")
        
        # Train with progress bar
        with click.progressbar(length=epochs, label=f'Training {gan_type} GAN') as bar:
            def progress_callback(epoch, g_loss, d_loss):
                if epoch % (epochs // 100 + 1) == 0:
                    bar.update(1)
                    bar.label = f'Epoch {epoch}/{epochs} - G: {g_loss:.4f}, D: {d_loss:.4f}'
                    
            # Start training
            history = gan.train(
                real_attacks=training_data,
                epochs=epochs,
                batch_size=batch_size
            )
            
            bar.update(epochs)
            
        # Save trained model
        model_path = output_path / f"{gan_type}_gan_model"
        gan.save_model(model_path)
        
        click.echo(f"✅ Training completed!")
        click.echo(f"💾 Model saved to {model_path}")
        click.echo(f"📈 Final G Loss: {history['g_loss'][-1]:.4f}")
        click.echo(f"📈 Final D Loss: {history['d_loss'][-1]:.4f}")
        
        # Generate sample outputs
        click.echo("🎯 Generating sample outputs...")
        
        if gan_type == 'attack':
            samples = gan.generate(num_samples=10)
            click.echo(f"Generated {len(samples)} attack samples")
        elif gan_type == 'malware':
            samples = gan.generate_malware_samples(num_samples=10)
            click.echo(f"Generated {len(samples)} malware samples")
        elif gan_type == 'network':
            samples = gan.generate_attack_patterns(num_patterns=10)
            click.echo(f"Generated {len(samples)} network attack patterns")
        elif gan_type == 'web':
            samples = gan.generate_web_attacks(num_attacks=10)
            click.echo(f"Generated {len(samples)} web attack sessions")
            
        # Save samples
        samples_file = output_path / f"{gan_type}_samples.json"
        _save_samples(samples, samples_file)
        click.echo(f"📄 Sample outputs saved to {samples_file}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        click.echo(f"❌ Training failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--range-info', '-r', type=click.Path(exists=True),
              help='Range info JSON file from deployment')
@click.option('--scenario', '-s', type=click.Choice(['basic_apt', 'ransomware', 'phishing', 'lateral_movement']),
              required=True, help='Attack scenario to execute')
@click.option('--intensity', '-i', type=click.Choice(['low', 'medium', 'high']),
              default='medium', help='Attack intensity')
@click.option('--duration', '-d', type=str, default='1h',
              help='Scenario duration (e.g., 30m, 2h, 1d)')
@click.option('--blue-team/--no-blue-team', default=True,
              help='Enable blue team defenses')
@click.option('--output-dir', '-o', type=click.Path(), default='./results',
              help='Output directory for results')
@click.pass_context
def scenario(ctx, range_info: Optional[str], scenario: str, intensity: str, 
             duration: str, blue_team: bool, output_dir: str):
    """Execute attack scenarios in cyber range."""
    
    try:
        logger.info(f"Executing {scenario} scenario with {intensity} intensity")
        
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load range info if provided
        if range_info:
            with open(range_info, 'r') as f:
                range_data = json.load(f)
            click.echo(f"📋 Using range: {range_data['name']}")
        else:
            click.echo("⚠️  No range info provided, using default configuration")
            
        # Create scenario configuration
        scenario_config = _create_scenario_config(scenario, intensity, duration)
        
        click.echo(f"🎯 Scenario: {scenario_config['name']}")
        click.echo(f"⚡ Intensity: {intensity}")
        click.echo(f"⏱️  Duration: {duration}")
        click.echo(f"🛡️  Blue Team: {'Enabled' if blue_team else 'Disabled'}")
        
        # Execute scenario
        with click.progressbar(length=100, label='Executing scenario') as bar:
            # Simulate scenario execution
            results = _execute_scenario(scenario_config, blue_team, bar)
            
        # Display results
        click.echo("📊 Scenario Results:")
        click.echo(f"  🎯 Attacks Executed: {results['attacks_executed']}")
        click.echo(f"  ✅ Successful Attacks: {results['successful_attacks']}")
        click.echo(f"  🚨 Detections: {results['detections']}")
        click.echo(f"  📈 Success Rate: {results['success_rate']:.1%}")
        click.echo(f"  🔍 Detection Rate: {results['detection_rate']:.1%}")
        
        # Save results
        results_file = output_path / f"{scenario}_{intensity}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        click.echo(f"💾 Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Scenario execution failed: {e}")
        click.echo(f"❌ Scenario failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'html']),
              default='json', help='Report format')
@click.option('--results-dir', '-r', type=click.Path(exists=True),
              help='Results directory to analyze')
@click.option('--output', '-o', type=click.Path(),
              help='Output report file')
@click.pass_context
def report(ctx, format: str, results_dir: Optional[str], output: Optional[str]):
    """Generate analysis reports from scenario results."""
    
    try:
        logger.info(f"Generating {format} report")
        
        if not results_dir:
            results_dir = './results'
            
        # Load all result files
        results_path = Path(results_dir)
        result_files = list(results_path.glob('*_results.json'))
        
        if not result_files:
            click.echo(f"⚠️  No result files found in {results_dir}")
            return
            
        click.echo(f"📁 Found {len(result_files)} result files")
        
        # Generate report
        report_data = _generate_analysis_report(result_files)
        
        # Format output
        if format == 'json':
            report_content = json.dumps(report_data, indent=2)
        elif format == 'csv':
            report_content = _format_csv_report(report_data)
        elif format == 'html':
            report_content = _format_html_report(report_data)
        else:
            raise ValueError(f"Unknown format: {format}")
            
        # Save or display report
        if output:
            with open(output, 'w') as f:
                f.write(report_content)
            click.echo(f"📄 Report saved to {output}")
        else:
            click.echo(report_content)
            
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        click.echo(f"❌ Report generation failed: {e}", err=True)
        raise click.Abort()


def _load_training_data(data_path: str, gan_type: str):
    """Load training data for GAN"""
    # Implementation would depend on data format and GAN type
    # For now, return dummy data
    return [f"sample_attack_{i}" for i in range(1000)]


def _save_samples(samples, file_path: Path):
    """Save generated samples to file"""
    # Convert samples to serializable format
    serializable_samples = []
    
    for sample in samples:
        if hasattr(sample, 'to_dict'):
            serializable_samples.append(sample.to_dict())
        elif hasattr(sample, '__dict__'):
            serializable_samples.append(sample.__dict__)
        else:
            serializable_samples.append(str(sample))
            
    with open(file_path, 'w') as f:
        json.dump(serializable_samples, f, indent=2, default=str)


def _create_scenario_config(scenario: str, intensity: str, duration: str) -> Dict[str, Any]:
    """Create scenario configuration"""
    
    scenario_configs = {
        'basic_apt': {
            'name': 'Basic APT Simulation',
            'description': 'Multi-stage APT attack simulation',
            'phases': ['reconnaissance', 'initial_access', 'persistence', 'lateral_movement', 'exfiltration'],
            'techniques': ['T1595', 'T1566', 'T1547', 'T1021', 'T1041']
        },
        'ransomware': {
            'name': 'Ransomware Attack',
            'description': 'Ransomware deployment and encryption',
            'phases': ['initial_access', 'execution', 'persistence', 'impact'],
            'techniques': ['T1566', 'T1059', 'T1547', 'T1486']
        },
        'phishing': {
            'name': 'Phishing Campaign',
            'description': 'Social engineering and credential harvesting',
            'phases': ['reconnaissance', 'delivery', 'exploitation'],
            'techniques': ['T1598', 'T1566', 'T1078']
        },
        'lateral_movement': {
            'name': 'Lateral Movement',
            'description': 'Internal network traversal',
            'phases': ['discovery', 'lateral_movement', 'collection'],
            'techniques': ['T1046', 'T1021', 'T1083']
        }
    }
    
    config = scenario_configs.get(scenario, scenario_configs['basic_apt'])
    config['intensity'] = intensity
    config['duration'] = duration
    
    return config


def _execute_scenario(config: Dict[str, Any], blue_team: bool, progress_bar) -> Dict[str, Any]:
    """Execute attack scenario"""
    
    import time
    import random
    
    # Simulate scenario execution
    total_attacks = random.randint(10, 50)
    successful_attacks = 0
    detections = 0
    
    for i in range(total_attacks):
        # Simulate attack
        time.sleep(0.1)  # Simulate work
        
        attack_success = random.random() < 0.7  # 70% success rate
        if attack_success:
            successful_attacks += 1
            
        # Simulate detection
        if blue_team and attack_success:
            detection = random.random() < 0.6  # 60% detection rate
            if detection:
                detections += 1
                
        # Update progress
        progress = int((i + 1) / total_attacks * 100)
        progress_bar.update(progress - progress_bar.pos)
        
    return {
        'scenario': config['name'],
        'attacks_executed': total_attacks,
        'successful_attacks': successful_attacks,
        'detections': detections,
        'success_rate': successful_attacks / total_attacks if total_attacks > 0 else 0,
        'detection_rate': detections / successful_attacks if successful_attacks > 0 else 0,
        'blue_team_enabled': blue_team,
        'timestamp': time.time()
    }


def _generate_analysis_report(result_files) -> Dict[str, Any]:
    """Generate analysis report from result files"""
    
    all_results = []
    
    for file_path in result_files:
        with open(file_path, 'r') as f:
            result = json.load(f)
            all_results.append(result)
            
    # Calculate aggregate metrics
    total_attacks = sum(r['attacks_executed'] for r in all_results)
    total_successful = sum(r['successful_attacks'] for r in all_results)
    total_detections = sum(r['detections'] for r in all_results)
    
    avg_success_rate = sum(r['success_rate'] for r in all_results) / len(all_results)
    avg_detection_rate = sum(r['detection_rate'] for r in all_results) / len(all_results)
    
    return {
        'summary': {
            'total_scenarios': len(all_results),
            'total_attacks': total_attacks,
            'total_successful_attacks': total_successful,
            'total_detections': total_detections,
            'average_success_rate': avg_success_rate,
            'average_detection_rate': avg_detection_rate
        },
        'scenarios': all_results
    }


def _format_csv_report(report_data: Dict[str, Any]) -> str:
    """Format report as CSV"""
    
    lines = ['Scenario,Attacks,Successful,Detections,Success Rate,Detection Rate']
    
    for scenario in report_data['scenarios']:
        line = f"{scenario['scenario']},{scenario['attacks_executed']},{scenario['successful_attacks']},{scenario['detections']},{scenario['success_rate']:.3f},{scenario['detection_rate']:.3f}"
        lines.append(line)
        
    return '\n'.join(lines)


def _format_html_report(report_data: Dict[str, Any]) -> str:
    """Format report as HTML"""
    
    html = """
    <html>
    <head><title>Cyber Range Report</title></head>
    <body>
    <h1>Cyber Range Analysis Report</h1>
    <h2>Summary</h2>
    <table border="1">
    <tr><td>Total Scenarios</td><td>{total_scenarios}</td></tr>
    <tr><td>Total Attacks</td><td>{total_attacks}</td></tr>
    <tr><td>Average Success Rate</td><td>{avg_success_rate:.1%}</td></tr>
    <tr><td>Average Detection Rate</td><td>{avg_detection_rate:.1%}</td></tr>
    </table>
    </body>
    </html>
    """.format(**report_data['summary'])
    
    return html


def main():
    """Main CLI entry point"""
    cli()


if __name__ == '__main__':
    main()