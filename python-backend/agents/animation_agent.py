"""
Animation Agent for Flutter Development
Handles animation implementation, performance optimization, and interactive animations
"""

from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent
from core.agent_types import AgentType

class AnimationAgent(BaseAgent):
    def __init__(self, agent_id: str = "animation_agent"):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.ANIMATION,
            name="Animation Agent",
            description="Specialized agent for Flutter animations, transitions, and interactive effects"
        )
        
        self.capabilities = [
            "implicit_animations",
            "explicit_animations", 
            "hero_animations",
            "custom_animations",
            "animation_controllers",
            "tween_animations",
            "staggered_animations",
            "physics_simulations",
            "interactive_animations",
            "performance_optimization",
            "animation_sequences",
            "lottie_animations"
        ]
        
        self.animation_types = {
            "implicit": ["AnimatedContainer", "AnimatedOpacity", "AnimatedPositioned", "AnimatedAlign"],
            "explicit": ["AnimationController", "Tween", "CurvedAnimation", "AnimatedBuilder"],
            "hero": ["Hero", "PageRouteBuilder", "CustomPageRoute"],
            "custom": ["CustomPainter", "CustomClipper", "Transform", "Matrix4"],
            "physics": ["SpringSimulation", "GravitySimulation", "FrictionSimulation"],
            "interactive": ["GestureDetector", "Draggable", "DragTarget", "Dismissible"]
        }

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process animation-related requests"""
        try:
            request_type = request.get('type', '')
            
            if request_type == 'create_animation':
                return await self._create_animation(request)
            elif request_type == 'optimize_animation':
                return await self._optimize_animation(request)
            elif request_type == 'create_transition':
                return await self._create_transition(request)
            elif request_type == 'implement_hero_animation':
                return await self._implement_hero_animation(request)
            elif request_type == 'create_custom_animation':
                return await self._create_custom_animation(request)
            elif request_type == 'add_lottie_animation':
                return await self._add_lottie_animation(request)
            else:
                return await self._analyze_animation_requirements(request)
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Animation processing failed: {str(e)}",
                'agent_id': self.agent_id
            }

    async def _create_animation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create Flutter animation implementation"""
        animation_type = request.get('animation_type', 'implicit')
        widget_name = request.get('widget_name', 'AnimatedWidget')
        properties = request.get('properties', [])
        duration = request.get('duration', 300)
        curve = request.get('curve', 'easeInOut')
        
        if animation_type == 'implicit':
            code = self._generate_implicit_animation(widget_name, properties, duration, curve)
        elif animation_type == 'explicit':
            code = self._generate_explicit_animation(widget_name, properties, duration, curve)
        else:
            code = self._generate_custom_animation(widget_name, properties, duration, curve)
        
        return {
            'success': True,
            'animation_code': code,
            'animation_type': animation_type,
            'duration': duration,
            'curve': curve,
            'agent_id': self.agent_id
        }

    async def _optimize_animation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize animation performance"""
        animation_code = request.get('code', '')
        optimization_type = request.get('optimization_type', 'performance')
        
        optimizations = []
        recommendations = []
        
        # Check for performance issues
        if 'setState' in animation_code and 'AnimationController' in animation_code:
            optimizations.append("Use AnimatedBuilder instead of setState for better performance")
            recommendations.append("Separate animation logic from widget rebuilds")
        
        if 'Transform' in animation_code:
            optimizations.append("Consider using Transform.translate for better GPU acceleration")
            recommendations.append("Use const constructors where possible")
        
        optimized_code = self._apply_animation_optimizations(animation_code, optimizations)
        
        return {
            'success': True,
            'optimized_code': optimized_code,
            'optimizations': optimizations,
            'recommendations': recommendations,
            'performance_tips': self._get_performance_tips(),
            'agent_id': self.agent_id
        }

    async def _create_transition(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create page transition animations"""
        transition_type = request.get('transition_type', 'slide')
        direction = request.get('direction', 'left_to_right')
        duration = request.get('duration', 300)
        
        transition_code = self._generate_page_transition(transition_type, direction, duration)
        
        return {
            'success': True,
            'transition_code': transition_code,
            'transition_type': transition_type,
            'direction': direction,
            'usage_example': self._get_transition_usage_example(),
            'agent_id': self.agent_id
        }

    async def _implement_hero_animation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Implement Hero animations for shared element transitions"""
        hero_tag = request.get('hero_tag', 'hero')
        widget_type = request.get('widget_type', 'Image')
        from_screen = request.get('from_screen', 'ListScreen')
        to_screen = request.get('to_screen', 'DetailScreen')
        
        hero_code = self._generate_hero_animation(hero_tag, widget_type, from_screen, to_screen)
        
        return {
            'success': True,
            'hero_code': hero_code,
            'hero_tag': hero_tag,
            'implementation_steps': self._get_hero_implementation_steps(),
            'best_practices': self._get_hero_best_practices(),
            'agent_id': self.agent_id
        }

    async def _create_custom_animation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create custom animations with AnimationController"""
        animation_name = request.get('name', 'CustomAnimation')
        properties = request.get('properties', ['opacity', 'scale'])
        duration = request.get('duration', 1000)
        repeat = request.get('repeat', False)
        
        custom_code = self._generate_custom_animation_controller(
            animation_name, properties, duration, repeat
        )
        
        return {
            'success': True,
            'custom_animation_code': custom_code,
            'controller_setup': self._get_controller_setup_code(),
            'disposal_code': self._get_disposal_code(),
            'usage_example': self._get_custom_animation_usage(),
            'agent_id': self.agent_id
        }

    async def _add_lottie_animation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Add Lottie animation support"""
        animation_path = request.get('path', 'assets/animations/loading.json')
        controller_type = request.get('controller_type', 'auto_play')
        
        lottie_code = self._generate_lottie_implementation(animation_path, controller_type)
        dependencies = ['lottie: ^2.7.0']
        
        return {
            'success': True,
            'lottie_code': lottie_code,
            'dependencies': dependencies,
            'asset_configuration': self._get_lottie_asset_config(),
            'controller_methods': self._get_lottie_controller_methods(),
            'agent_id': self.agent_id
        }

    def _generate_implicit_animation(self, widget_name: str, properties: List[str], 
                                   duration: int, curve: str) -> str:
        """Generate implicit animation code"""
        return f'''
class {widget_name} extends StatefulWidget {{
  @override
  _{widget_name}State createState() => _{widget_name}State();
}}

class _{widget_name}State extends State<{widget_name}> {{
  bool _isExpanded = false;
  
  @override
  Widget build(BuildContext context) {{
    return GestureDetector(
      onTap: () {{
        setState(() {{
          _isExpanded = !_isExpanded;
        }});
      }},
      child: AnimatedContainer(
        duration: Duration(milliseconds: {duration}),
        curve: Curves.{curve},
        width: _isExpanded ? 200.0 : 100.0,
        height: _isExpanded ? 200.0 : 100.0,
        decoration: BoxDecoration(
          color: _isExpanded ? Colors.blue : Colors.red,
          borderRadius: BorderRadius.circular(_isExpanded ? 20.0 : 10.0),
        ),
        child: Center(
          child: Text(
            'Tap me!',
            style: TextStyle(
              color: Colors.white,
              fontSize: _isExpanded ? 18.0 : 14.0,
            ),
          ),
        ),
      ),
    );
  }}
}}
'''

    def _generate_explicit_animation(self, widget_name: str, properties: List[str], 
                                   duration: int, curve: str) -> str:
        """Generate explicit animation with AnimationController"""
        return f'''
class {widget_name} extends StatefulWidget {{
  @override
  _{widget_name}State createState() => _{widget_name}State();
}}

class _{widget_name}State extends State<{widget_name}> 
    with SingleTickerProviderStateMixin {{
  late AnimationController _controller;
  late Animation<double> _animation;
  
  @override
  void initState() {{
    super.initState();
    _controller = AnimationController(
      duration: Duration(milliseconds: {duration}),
      vsync: this,
    );
    _animation = CurvedAnimation(
      parent: _controller,
      curve: Curves.{curve},
    );
  }}
  
  @override
  void dispose() {{
    _controller.dispose();
    super.dispose();
  }}
  
  @override
  Widget build(BuildContext context) {{
    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {{
        return Transform.scale(
          scale: 1.0 + (_animation.value * 0.5),
          child: Opacity(
            opacity: _animation.value,
            child: Container(
              width: 100,
              height: 100,
              decoration: BoxDecoration(
                color: Colors.blue,
                borderRadius: BorderRadius.circular(10),
              ),
              child: child,
            ),
          ),
        );
      }},
      child: Icon(Icons.star, color: Colors.white),
    );
  }}
  
  void startAnimation() {{
    _controller.forward();
  }}
  
  void reverseAnimation() {{
    _controller.reverse();
  }}
}}
'''

    def _generate_page_transition(self, transition_type: str, direction: str, duration: int) -> str:
        """Generate page transition animation"""
        return f'''
class CustomPageRoute<T> extends PageRouteBuilder<T> {{
  final Widget child;
  
  CustomPageRoute({{required this.child}})
      : super(
          pageBuilder: (context, animation, secondaryAnimation) => child,
          transitionDuration: Duration(milliseconds: {duration}),
          transitionsBuilder: (context, animation, secondaryAnimation, child) {{
            return _build{transition_type.capitalize()}Transition(
              animation, 
              child, 
              direction: '{direction}'
            );
          }},
        );
  
  static Widget _build{transition_type.capitalize()}Transition(
    Animation<double> animation, 
    Widget child, 
    {{required String direction}}
  ) {{
    Offset begin;
    switch (direction) {{
      case 'left_to_right':
        begin = Offset(-1.0, 0.0);
        break;
      case 'right_to_left':
        begin = Offset(1.0, 0.0);
        break;
      case 'top_to_bottom':
        begin = Offset(0.0, -1.0);
        break;
      case 'bottom_to_top':
        begin = Offset(0.0, 1.0);
        break;
      default:
        begin = Offset(-1.0, 0.0);
    }}
    
    return SlideTransition(
      position: Tween<Offset>(
        begin: begin,
        end: Offset.zero,
      ).animate(CurvedAnimation(
        parent: animation,
        curve: Curves.easeInOut,
      )),
      child: child,
    );
  }}
}}

// Usage example:
// Navigator.push(
//   context,
//   CustomPageRoute(child: NextScreen()),
// );
'''

    def _generate_hero_animation(self, hero_tag: str, widget_type: str, 
                               from_screen: str, to_screen: str) -> str:
        """Generate Hero animation implementation"""
        return f'''
// {from_screen} - Source screen
class {from_screen} extends StatelessWidget {{
  @override
  Widget build(BuildContext context) {{
    return Scaffold(
      body: GestureDetector(
        onTap: () {{
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => {to_screen}()),
          );
        }},
        child: Hero(
          tag: '{hero_tag}',
          child: {widget_type}(
            width: 100,
            height: 100,
            // Add your image or widget properties here
          ),
        ),
      ),
    );
  }}
}}

// {to_screen} - Destination screen
class {to_screen} extends StatelessWidget {{
  @override
  Widget build(BuildContext context) {{
    return Scaffold(
      appBar: AppBar(title: Text('Detail View')),
      body: Center(
        child: Hero(
          tag: '{hero_tag}',
          child: {widget_type}(
            width: 300,
            height: 300,
            // Same widget type with different properties
          ),
        ),
      ),
    );
  }}
}}
'''

    def _generate_lottie_implementation(self, animation_path: str, controller_type: str) -> str:
        """Generate Lottie animation implementation"""
        return f'''
import 'package:lottie/lottie.dart';

class LottieAnimationWidget extends StatefulWidget {{
  @override
  _LottieAnimationWidgetState createState() => _LottieAnimationWidgetState();
}}

class _LottieAnimationWidgetState extends State<LottieAnimationWidget>
    with TickerProviderStateMixin {{
  late AnimationController _lottieController;
  
  @override
  void initState() {{
    super.initState();
    _lottieController = AnimationController(vsync: this);
  }}
  
  @override
  void dispose() {{
    _lottieController.dispose();
    super.dispose();
  }}
  
  @override
  Widget build(BuildContext context) {{
    return Column(
      children: [
        Lottie.asset(
          '{animation_path}',
          controller: _lottieController,
          onLoaded: (composition) {{
            _lottieController.duration = composition.duration;
            {'_lottieController.repeat();' if controller_type == 'auto_play' else ''}
          }},
        ),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () => _lottieController.forward(),
              child: Text('Play'),
            ),
            SizedBox(width: 8),
            ElevatedButton(
              onPressed: () => _lottieController.stop(),
              child: Text('Stop'),
            ),
            SizedBox(width: 8),
            ElevatedButton(
              onPressed: () => _lottieController.reset(),
              child: Text('Reset'),
            ),
          ],
        ),
      ],
    );
  }}
}}
'''

    def _get_performance_tips(self) -> List[str]:
        """Get animation performance optimization tips"""
        return [
            "Use const constructors for widgets that don't change",
            "Prefer AnimatedBuilder over setState for animations",
            "Use Transform widgets for GPU-accelerated animations",
            "Avoid rebuilding expensive widgets during animations",
            "Use RepaintBoundary to isolate animation layers",
            "Consider using Opacity widget instead of AnimatedOpacity for simple cases",
            "Use SingleTickerProviderStateMixin for single animations",
            "Dispose animation controllers to prevent memory leaks"
        ]

    def _get_hero_best_practices(self) -> List[str]:
        """Get Hero animation best practices"""
        return [
            "Use unique hero tags to avoid conflicts",
            "Ensure hero widgets have similar aspect ratios",
            "Consider using custom flight shuttle builders for complex transitions",
            "Test hero animations with different screen sizes",
            "Handle hero animation interruptions gracefully"
        ]

    async def _analyze_animation_requirements(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze animation requirements and suggest implementation"""
        description = request.get('description', '')
        complexity = request.get('complexity', 'medium')
        
        suggestions = self._get_animation_suggestions(description, complexity)
        
        return {
            'success': True,
            'suggestions': suggestions,
            'recommended_approach': self._recommend_animation_approach(description),
            'complexity_analysis': self._analyze_complexity(complexity),
            'implementation_steps': self._get_implementation_steps(description),
            'agent_id': self.agent_id
        }

    def _get_animation_suggestions(self, description: str, complexity: str) -> List[str]:
        """Get animation implementation suggestions"""
        suggestions = []
        
        if 'transition' in description.lower():
            suggestions.append("Consider using PageRouteBuilder for custom page transitions")
        if 'loading' in description.lower():
            suggestions.append("Use CircularProgressIndicator or Lottie animations for loading states")
        if 'gesture' in description.lower():
            suggestions.append("Implement GestureDetector with custom animations")
        if 'list' in description.lower():
            suggestions.append("Use AnimatedList for dynamic list animations")
            
        return suggestions

    def _recommend_animation_approach(self, description: str) -> str:
        """Recommend the best animation approach"""
        if any(word in description.lower() for word in ['simple', 'basic', 'fade', 'scale']):
            return "implicit_animations"
        elif any(word in description.lower() for word in ['complex', 'custom', 'physics', 'interactive']):
            return "explicit_animations"
        elif 'hero' in description.lower() or 'shared element' in description.lower():
            return "hero_animations"
        else:
            return "implicit_animations"
